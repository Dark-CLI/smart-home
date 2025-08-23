from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from .const import DOMAIN
from .llm_api import MyAPI


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    unreg = llm.async_register_api(
        hass, MyAPI(hass, f"{DOMAIN}-{entry.entry_id}", "My LLM Service API")
    )
    entry.async_on_unload(unreg)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    return True
