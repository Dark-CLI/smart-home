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
    """Processes tool calls from the language model.

    This class is responsible for parsing the language model's response,
    identifying tool calls, validating them, and executing them. It can handle
    both standard Home Assistant services and custom-defined tools.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        llm_api: llm.APIInstance,
        entry: ConfigEntry,
    ):
        """Initializes the ToolProcessor.

        Args:
            hass: The Home Assistant instance.
            llm_api: The language model API instance.
            entry: The config entry for the integration.
        """
        self.hass = hass
        self.llm_api = llm_api
        self.entry = entry

        service_call_regex = self.entry.options.get(
            CONF_SERVICE_CALL_REGEX, DEFAULT_SERVICE_CALL_REGEX
        )
        self.service_call_pattern = re.compile(service_call_regex, flags=re.MULTILINE)

        self._custom_tool_handlers: dict[str, Callable] = {
            "HassSetNumber": self._handle_number_set_value,
            "HassPressButton": self._handle_button_press,
        }

    async def async_process_tool_calls(
        self, response: str
    ) -> tuple[str, list[dict[str, Any]]]:
        """Processes the language model's response for tool calls.

        This method searches for tool call blocks in the response, parses them,
        validates them, and executes them. It returns the part of the response
        that is meant to be spoken to the user, and a list of tool call results
        that should be sent back to the model.

        Args:
            response: The raw response from the language model.

        Returns:
            A tuple containing:
            - The part of the response to be spoken to the user.
            - A list of tool call results to be sent back to the model.
        """
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
                tool_results_to_send.append(
                    {
                        "tool_call": block,
                        "result": {
                            "error": "invalid_json",
                            "error_text": "The tool call was not valid JSON.",
                        },
                    }
                )
            except vol.Error as ex:
                _LOGGER.info(
                    f"LLM produced an improperly formatted response: {repr(ex)}"
                )
                tool_results_to_send.append(
                    {
                        "tool_call": parsed_tool_call if parsed_tool_call else block,
                        "result": {"error": "validation_error", "error_text": str(ex)},
                    }
                )
            except HomeAssistantError as e:
                _LOGGER.debug("Tool execution error: %s", e)
                tool_results_to_send.append(
                    {
                        "tool_call": parsed_tool_call if parsed_tool_call else block,
                        "result": {"error": type(e).__name__, "error_text": str(e)},
                    }
                )
            except Exception as e:
                _LOGGER.exception("An unexpected error occurred during tool processing")
                tool_results_to_send.append(
                    {
                        "tool_call": parsed_tool_call if parsed_tool_call else block,
                        "result": {"error": type(e).__name__, "error_text": str(e)},
                    }
                )

        return to_say, tool_results_to_send

    def _parse_tool_call(self, block: str) -> dict:
        """
        Parses a tool call block from the LLM response.

        Args:
            block: The JSON string representing the tool call.

        Returns:
            The parsed tool call as a dictionary.
        """
        return json.loads(block)

    def _validate_tool_call(self, parsed_tool_call: dict) -> None:
        """
        Validates the structure of a parsed tool call.

        Args:
            parsed_tool_call: The tool call to validate.

        Raises:
            vol.Error: If the tool call is not valid.
        """
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
        """
        Fixes tool arguments to be compatible with Home Assistant services.

        This method performs conversions, such as brightness from a 0-1 float
        to a 0-255 integer, and string-to-list conversion for colors.

        Args:
            args_dict: The dictionary of arguments for the tool call.

        Returns:
            The fixed dictionary of arguments.
        """
        if "brightness" in args_dict and 0.0 < args_dict["brightness"] <= 1.0:
            args_dict["brightness"] = int(args_dict["brightness"] * 255)

        if "rgb_color" in args_dict and isinstance(args_dict["rgb_color"], str):
            args_dict["rgb_color"] = [
                int(x) for x in args_dict["rgb_color"][1:-1].split(",")
            ]
        return args_dict

    async def _execute_tool_call(self, tool_input: llm.ToolInput) -> Any:
        """Executes a tool call.

        This method checks for custom tool handlers first, and if none are
        found, it passes the tool call to the llm_api.

        Args:
            tool_input: The tool input to execute.

        Returns:
            The result of the tool call.
        """
        if tool_input.tool_name in self._custom_tool_handlers:
            handler = self._custom_tool_handlers[tool_input.tool_name]
            return await handler(tool_input.tool_args)
        return await self.llm_api.async_call_tool(tool_input)

    async def _handle_number_set_value(self, args: dict) -> dict:
        """Handles the custom HassSetNumber tool call.

        This tool allows the LLM to set the value of a number entity. It supports
        targeting by entity, device, area, label, and floor.

        Args:
            args: The arguments for the tool call. Must include 'value' and at
                  least one target specifier (e.g., 'name', 'area', 'device_id').

                  Examples:
                    - {"name": "input_number.temperature_setpoint", "value": 22}
                    - {"area": "Living Room", "value": 23}
                    - {"device_id": "abcdef123456", "value": 21.5}

        Returns:
            A dictionary indicating success or failure.
        """
        target = {}
        service_data = {}

        value = args.get("value")
        if value is None:
            return {"success": False, "error": "Missing value"}

        service_data["value"] = value

        # Map LLM arguments to target keys
        target_mapping = {
            "name": "entity_id",
            "entity_id": "entity_id",
            "device_id": "device_id",
            "area": "area_id",
            "area_id": "area_id",
            "label": "label_id",
            "label_id": "label_id",
            "floor": "floor_id",
            "floor_id": "floor_id",
        }

        for arg_key, target_key in target_mapping.items():
            if arg_key in args:
                target[target_key] = args[arg_key]

        if not target:
            return {
                "success": False,
                "error": "Missing target. Please specify at least one of: name, entity_id, device_id, area, label, or floor.",
            }

        try:
            await self.hass.services.async_call(
                "number",
                "set_value",
                service_data=service_data,
                target=target,
                blocking=True,
            )
            return {"success": True, "target": target, "value": value}
        except Exception as e:
            _LOGGER.error(f"Error setting number value for target {target}: {e}")
            return {"success": False, "error": str(e), "target": target}

    async def _handle_button_press(self, args: dict) -> dict:
        """Handles the custom HassPressButton tool call.

        This tool allows the LLM to press a button. It supports
        targeting by entity, device, area, label, and floor.

        Args:
            args: The arguments for the tool call. Must include at least one
                  target specifier (e.g., 'name', 'area', 'device_id').

                  Examples:
                    - {"name": "button.restart_my_pc"}
                    - {"area": "Living Room"}
                    - {"device_id": "abcdef123456"}
        """
        target = {}

        # Map LLM arguments to target keys
        target_mapping = {
            "name": "entity_id",
            "entity_id": "entity_id",
            "device_id": "device_id",
            "area": "area_id",
            "area_id": "area_id",
            "label": "label_id",
            "label_id": "label_id",
            "floor": "floor_id",
            "floor_id": "floor_id",
        }

        for arg_key, target_key in target_mapping.items():
            if arg_key in args:
                target[target_key] = args[arg_key]

        if not target:
            return {
                "success": False,
                "error": "Missing target. Please specify at least one of: name, entity_id, device_id, area, label, or floor.",
            }

        try:
            await self.hass.services.async_call(
                "button",
                "press",
                service_data={},
                target=target,
                blocking=True,
            )
            return {"success": True, "target": target}
        except Exception as e:
            _LOGGER.error(f"Error pressing button for target {target}: {e}")
            return {"success": False, "error": str(e), "target": target}
