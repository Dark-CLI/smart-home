from __future__ import annotations

import json
import logging
import re
from typing import Any

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

    async def async_process_tool_calls(
        self, response: str
    ) -> tuple[str, list[dict[str, Any]]]:
        to_say = self.service_call_pattern.sub("", response.strip())
        tool_calls = self.service_call_pattern.findall(response.strip())
        tool_results_to_send = []

        if not tool_calls:
            return to_say, []

        for block in tool_calls:
            try:
                parsed_tool_call: dict = json.loads(block)
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
                continue

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

            try:
                schema_to_validate(parsed_tool_call)
            except vol.Error as ex:
                _LOGGER.info(
                    f"LLM produced an improperly formatted response: {repr(ex)}"
                )
                tool_results_to_send.append(
                    {
                        "tool_call": parsed_tool_call,
                        "result": {"error": "validation_error", "error_text": str(ex)},
                    }
                )
                continue

            _LOGGER.info(f"calling tool: {block}")

            args_dict = (
                parsed_tool_call
                if self.llm_api.api.id == HOME_LLM_API_ID
                else parsed_tool_call["arguments"]
            )

            if "brightness" in args_dict and 0.0 < args_dict["brightness"] <= 1.0:
                args_dict["brightness"] = int(args_dict["brightness"] * 255)

            if "rgb_color" in args_dict and isinstance(args_dict["rgb_color"], str):
                args_dict["rgb_color"] = [
                    int(x) for x in args_dict["rgb_color"][1:-1].split(",")
                ]

            if self.llm_api.api.id == HOME_LLM_API_ID:
                to_say = to_say + parsed_tool_call.pop("to_say", "")
                tool_input = llm.ToolInput(
                    tool_name=SERVICE_TOOL_NAME,
                    tool_args=parsed_tool_call,
                )
            else:
                tool_input = llm.ToolInput(
                    tool_name=parsed_tool_call["name"],
                    tool_args=parsed_tool_call["arguments"],
                )

            tool_response = None
            try:
                tool_response = await self.llm_api.async_call_tool(tool_input)
                _LOGGER.debug("Tool response: %s", tool_response)
                if tool_response and tool_response.get("result"):
                    tool_results_to_send.append(
                        {"tool_call": parsed_tool_call, "result": tool_response}
                    )
            except (HomeAssistantError, vol.Invalid) as e:
                tool_response = {"error": type(e).__name__}
                if str(e):
                    tool_response["error_text"] = str(e)
                _LOGGER.debug("Tool response: %s", tool_response)
                tool_results_to_send.append(
                    {"tool_call": parsed_tool_call, "result": tool_response}
                )

        return to_say, tool_results_to_send