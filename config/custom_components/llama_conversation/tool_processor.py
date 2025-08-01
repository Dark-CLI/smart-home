from __future__ import annotations

import json
import logging
import re
import asyncio
from typing import Any, Callable

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from .const import (
    CONF_SERVICE_CALL_REGEX,
    DEFAULT_SERVICE_CALL_REGEX,
    HOME_LLM_API_ID,
    SERVICE_TOOL_NAME,
)

_LOGGER = logging.getLogger(__name__)


class ToolProcessor:
    def __init__(
        self,
        hass: HomeAssistant,
        llm_api: llm.APIInstance,
        entry: ConfigEntry,
    ):
        self.hass = hass
        self.llm_api = llm_api
        self.entry = entry

        service_call_regex = self.entry.options.get(
            CONF_SERVICE_CALL_REGEX, DEFAULT_SERVICE_CALL_REGEX
        )
        self.service_call_pattern = re.compile(service_call_regex, flags=re.MULTILINE)

        self._custom_tool_handlers: dict[str, Callable] = {
            "HassSetNumber": self._handle_number_set_value,
        }

    async def async_process_tool_calls(
        self, response: str
    ) -> tuple[str, list[dict[str, Any]]]:
        to_say = self.service_call_pattern.sub("", response.strip())
        tool_calls_raw = self.service_call_pattern.findall(response.strip())
        tool_results_to_send = []

        if not tool_calls_raw:
            return to_say, []

        for block in tool_calls_raw:
            parsed_tool_call = None
            try:
                parsed_tool_call = self._parse_tool_call(block)
                self._validate_tool_call(parsed_tool_call)

                if self.llm_api.api.id == HOME_LLM_API_ID:
                    tool_name = SERVICE_TOOL_NAME
                    args_dict = parsed_tool_call
                else:
                    tool_name = parsed_tool_call["name"]
                    args_dict = parsed_tool_call["arguments"]

                fixed_args = self._fix_tool_arguments(args_dict)

                tool_input = llm.ToolInput(
                    tool_name=tool_name,
                    tool_args=fixed_args,
                )

                _LOGGER.info(f"Calling tool: {tool_name} with args: {fixed_args}")
                tool_response = await self._execute_tool_call(tool_input)

                _LOGGER.debug("Tool response: %s", tool_response)
                if tool_response and tool_response.get("result"):
                    tool_results_to_send.append(
                        {"tool_call": parsed_tool_call, "result": tool_response}
                    )

            except json.JSONDecodeError:
                _LOGGER.warning(f"LLM produced an invalid JSON tool call: {block}")
                tool_results_to_send.append({
                    "tool_call": block,
                    "result": {"error": "invalid_json", "error_text": "The tool call was not valid JSON."}
                })
            except vol.Error as ex:
                _LOGGER.info(
                    f"LLM produced an improperly formatted response: {repr(ex)}"
                )
                tool_results_to_send.append({
                    "tool_call": parsed_tool_call if parsed_tool_call else block,
                    "result": {"error": "validation_error", "error_text": str(ex)}
                })
            except HomeAssistantError as e:
                _LOGGER.debug("Tool execution error: %s", e)
                tool_results_to_send.append({
                    "tool_call": parsed_tool_call if parsed_tool_call else block,
                    "result": {"error": type(e).__name__, "error_text": str(e)}
                })
            except Exception as e:
                _LOGGER.exception("An unexpected error occurred during tool processing")
                tool_results_to_send.append({
                    "tool_call": parsed_tool_call if parsed_tool_call else block,
                    "result": {"error": type(e).__name__, "error_text": str(e)}
                })

        return to_say, tool_results_to_send

    def _parse_tool_call(self, block: str) -> dict:
        return json.loads(block)

    def _validate_tool_call(self, parsed_tool_call: dict) -> None:
        if self.llm_api.api.id == HOME_LLM_API_ID:
            schema_to_validate = vol.Schema(
                {
                    vol.Required("service"): str,
                    vol.Required("target_device"): str,
                    vol.Optional("rgb_color"): str,
                    vol.Optional("brightness"): vol.Coerce(float),
                    vol.Optional("temperature"): vol.Coerce(float),
                    vol.Optional("humidity"): vol.Coerce(float),
                    vol.Optional("fan_mode"): str,
                    vol.Optional("hvac_mode"): str,
                    vol.Optional("preset_mode"): str,
                    vol.Optional("duration"): str,
                    vol.Optional("item"): str,
                }
            )
        else:
            schema_to_validate = vol.Schema(
                {
                    vol.Required("name"): str,
                    vol.Required("arguments"): dict,
                }
            )
        schema_to_validate(parsed_tool_call)

    def _fix_tool_arguments(self, args_dict: dict) -> dict:
        if "brightness" in args_dict and 0.0 < args_dict["brightness"] <= 1.0:
            args_dict["brightness"] = int(args_dict["brightness"] * 255)

        if "rgb_color" in args_dict and isinstance(args_dict["rgb_color"], str):
            args_dict["rgb_color"] = [
                int(x) for x in args_dict["rgb_color"][1:-1].split(",")
            ]
        return args_dict

    async def _execute_tool_call(self, tool_input: llm.ToolInput) -> Any:
        if tool_input.tool_name in self._custom_tool_handlers:
            handler = self._custom_tool_handlers[tool_input.tool_name]
            return await handler(tool_input.tool_args)
        elif self.llm_api.api.id == HOME_LLM_API_ID:
            return await self.llm_api.async_call_tool(tool_input)
        else:
            return await self.llm_api.async_call_tool(tool_input)

    async def _handle_number_set_value(self, args: dict) -> dict:
        entity_id = args.get("name")
        value = args.get("value")

        if not entity_id or value is None:
            return {"success": False, "error": "Missing entity_id or value"}

        try:
            await self.hass.services.async_call(
                "number", "set_value", {"entity_id": entity_id, "value": value}, blocking=True
            )
            return {"success": True, "entity_id": entity_id, "value": value}
        except Exception as e:
            _LOGGER.error(f"Error setting number value for {entity_id}: {e}")
            return {"success": False, "error": str(e), "entity_id": entity_id}