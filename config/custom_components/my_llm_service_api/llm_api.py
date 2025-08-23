from __future__ import annotations
from typing import Any
import fnmatch
import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.llm import LLMContext, ToolInput

ALLOWED_DOMAINS = {"light", "switch", "climate", "fan", "number"}  # tighten this


class GetServiceSchema(llm.Tool):
    name = "GetServiceSchema"
    description = "List available services and parameter hints. Call this before calling a service."
    parameters = vol.Schema(
        {
            vol.Optional("domain"): str,
            vol.Optional("service"): str,
        }
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> dict[str, Any]:
        # TODO: Handle tool failure more gracefully instead of returning nothing, which can cause LLM hallucinations.
        domain = tool_input.tool_args.get("domain")
        service = tool_input.tool_args.get("service")
        out: dict[str, Any] = {}
        for dom, services in hass.services.async_services().items():
            if domain and dom != domain:
                continue
            if ALLOWED_DOMAINS and dom not in ALLOWED_DOMAINS:
                continue
            for srv, obj in services.items():
                if service and srv != service:
                    continue
                schema = getattr(obj, "schema", None)
                fields_hint = []
                # The schema can be a validator object (e.g., vol.All),
                # which doesn't have a .schema attribute. Only vol.Schema objects
                # have a .schema attribute that contains the schema dictionary.
                if isinstance(schema, vol.Schema) and isinstance(schema.schema, dict):
                    fields_hint = [k.schema for k in schema.schema]

                target_schema = getattr(obj, "target_schema", None)
                target_hint = []
                # The schema can be a validator object (e.g., vol.All),
                # which doesn't have a .schema attribute. Only vol.Schema objects
                # have a .schema attribute that contains the schema dictionary.
                if isinstance(target_schema, vol.Schema) and isinstance(
                    target_schema.schema, dict
                ):
                    target_hint = [k.schema for k in target_schema.schema]

                out.setdefault(dom, {})[srv] = {
                    "fields_hint": fields_hint,
                    "target_hint": target_hint,
                }

        if not out:
            raise HomeAssistantError("No matching services found.")
        return out


class CallService(llm.Tool):
    name = "CallService"
    description = (
        "Execute a Home Assistant service call. Use only after GetServiceSchema."
    )
    parameters = vol.Schema(
        {
            vol.Required("domain"): str,
            vol.Required("service"): str,
            vol.Optional("data", default=dict): dict,
            vol.Optional(
                "target", default=dict
            ): dict,  # {entity_id: "..."} or {area_id: "..."}
            vol.Optional("blocking", default=True): bool,
        }
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> dict[str, Any]:
        # TODO: Handle tool failure more gracefully instead of returning nothing, which can cause LLM hallucinations.
        dom = tool_input.tool_args["domain"]
        srv = tool_input.tool_args["service"]
        data = dict(tool_input.tool_args.get("data") or {})
        target = dict(tool_input.tool_args.get("target") or {})
        blocking = bool(tool_input.tool_args.get("blocking", True))

        print(f"Calling service: {dom}.{srv}")
        print(f"  Data: {data}")
        print(f"  Target: {target}")
        print(f"  Blocking: {blocking}")

        # if target:
        #     data = {**data, "target": target}

        if ALLOWED_DOMAINS and dom not in ALLOWED_DOMAINS:
            raise HomeAssistantError(f"Domain not allowed: {dom}")

        try:
            result = await hass.services.async_call(
                domain=dom,
                service=srv,
                service_data=data,
                target=target,
                blocking=blocking,
                context=llm_context.context,  # attribute to the convo user
            )
            print(f"  Result: {result}")
            return {"ok": True, "result": result}
        except Exception as e:
            print(f"  Error calling service: {e}")
            return {"ok": False, "error": str(e)}


class GetLiveContext(llm.Tool):
    name = "GetLiveContext"
    description = "Use this tool to find the `entity_id` for a device or to get the current state of entities. You can search by friendly name or other parts of the entity ID using the `entity_id_glob` parameter. For example, to find a 'dimmer light', you could use the glob `'*dimmer*'`."
    parameters = vol.Schema(
        {
            vol.Optional("entity_id_glob"): str,
        }
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext
    ) -> dict[str, Any]:
        # TODO: Handle tool failure more gracefully instead of returning nothing, which can cause LLM hallucinations.
        entity_id_glob = tool_input.tool_args.get("entity_id_glob")
        out: dict[str, Any] = {}

        for state in hass.states.async_all():
            if entity_id_glob and not fnmatch.fnmatch(state.entity_id, entity_id_glob):
                continue
            out[state.entity_id] = {
                "state": state.state,
                "attributes": dict(state.attributes),
            }

        if not out:
            raise HomeAssistantError("No matching entities found.")
        return out


class MyAPI(llm.API):
    def __init__(self, hass: HomeAssistant, id: str, name: str) -> None:
        """Initialize the API."""
        self.hass = hass
        self.id = id
        self.name = name

    async def async_get_api_instance(self, llm_context: LLMContext) -> llm.APIInstance:
        return llm.APIInstance(
            api=self,
            llm_context=llm_context,
            api_prompt=(
                "**System Rules:**\n"
                "- You are a voice assistant for Home Assistant.\n"
                "- Your primary goal is to control devices and report their state.\n"
                "- Be direct and to the point. Do not apologize or use conversational filler.\n"
                "**Tool Calling Rules:**\n"
                "- You MUST use the `tool_calls` JSON field to execute tools.\n"
                "- NEVER write a tool call as plain text in your response.\n\n"
                "**Protocol:**\n"
                "1) First, use GetLiveContext to find the `entity_id` of the target device.\n"
                "2) Next, use GetServiceSchema with only the `domain` of the entity you found. This will show you all available services for that device.\n"
                "3) From the list of services, choose the most appropriate one and identify the parameters you need.\n"
                "4) Finally, use CallService with the correct `domain`, `service`, `target`, and any `data` required.\n\n"
                "**Example:**\n"
                "User: 'Set the AC to 20 degrees.'\n"
                "Assistant Tool Call: GetLiveContext(entity_id_glob='*ac*')\n"
                'Tool Output: {"number.devices_room_ac_temperature": {"state": "16.0", ...}}\n'
                "Assistant Tool Call: GetServiceSchema(domain='number')\n"
                'Tool Output: {"number": {"set_value": {"fields_hint": ["value"], ...}}}\n'
                "Assistant Tool Call: CallService(domain='number', service='set_value', target={'entity_id': 'number.devices_room_ac_temperature'}, data={'value': 20})\n"
                'Tool Output: {"ok": true, "result": null}\n'
                "Assistant: 'Done. The AC temperature is set to 20.'\n"
            ),
            tools=[GetServiceSchema(), CallService(), GetLiveContext()],
        )
