"""Business context operator - fetches config from Django API.

This operator fetches business configuration from the Django API when a call
starts, populating the call context with business-specific data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx
from loguru import logger

from operators.base import Operator
from pipeline.events import CallContext, CallEvent, CallLifecycleEvent, OperatorOutput


@dataclass
class BusinessConfig:
    """Business configuration fetched from Django API."""

    business_id: str
    business_name: str
    industry: str
    timezone: str
    system_prompt: str
    persona: str
    style: str  # Communication style: friendly, direct, professional, empathetic
    voice_id: Optional[str]
    greeting_template: str
    closing_template: str
    services: list[str]
    service_area: list[str]
    business_description: str
    hours_schedule: Dict[str, Any]
    forward_urgent_calls: bool
    emergency_forward_number: str
    emergency_keywords: list[str]
    legal_disclaimer: str
    custom_call_flow: Optional[Dict[str, Any]]
    knowledge_docs: list[str]


class BusinessContextOperator(Operator):
    """Fetches business config from Django API on call start.

    This operator:
    1. Listens for "connected" lifecycle events
    2. Fetches business config from Django via internal API
    3. Updates the call context with business data
    4. Emits the system prompt for use by the LLM

    Configuration:
        Set DJANGO_INTERNAL_URL environment variable to the Django API URL.
        In production on Fly.io, this would be: http://clark-web.internal:8000

    Usage:
        operators = [
            BusinessContextOperator(),
            # ... other operators
        ]
    """

    name = "business_context"

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        timeout: float = 5.0,
    ) -> None:
        self._api_base_url = api_base_url or os.getenv(
            "DJANGO_INTERNAL_URL", "http://localhost:8000"
        )
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._config: Optional[BusinessConfig] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy initialization of HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._api_base_url,
                timeout=self._timeout,
            )
        return self._client

    def _parse_config_response(self, data: dict) -> BusinessConfig:
        """Parse API response into BusinessConfig dataclass."""
        return BusinessConfig(
            business_id=data["business_id"],
            business_name=data["business_name"],
            industry=data["industry"],
            timezone=data["timezone"],
            system_prompt=data["system_prompt"],
            persona=data["persona"],
            style=data.get("style", "friendly"),
            voice_id=data.get("voice_id"),
            greeting_template=data["greeting_template"],
            closing_template=data["closing_template"],
            services=data.get("services", []),
            service_area=data.get("service_area", []),
            business_description=data.get("business_description", ""),
            hours_schedule=data.get("hours_schedule", {}),
            forward_urgent_calls=data.get("forward_urgent_calls", False),
            emergency_forward_number=data.get("emergency_forward_number", ""),
            emergency_keywords=data.get("emergency_keywords", []),
            legal_disclaimer=data.get("legal_disclaimer", ""),
            custom_call_flow=data.get("custom_call_flow"),
            knowledge_docs=data.get("knowledge_docs", []),
        )

    def _load_demo_config(self, industry: str) -> Optional[dict]:
        """Load demo configuration from JSON file."""
        import json
        from pathlib import Path

        json_path = Path(__file__).parent.parent / "demo_personas.json"
        try:
            with open(json_path, "r") as f:
                personas = json.load(f)
                return personas.get(industry, personas.get("towing"))
        except Exception as e:
            logger.error(f"Failed to load demo personas: {e}")
            return None

    def _build_demo_system_prompt(self, config: dict) -> str:
        """Build a comprehensive system prompt for demo calls."""
        business_name = config.get("business_name", "Demo Business")
        industry = config.get("industry", "general")
        phone = config.get("phone", "")
        timezone = config.get("timezone", "America/New_York")
        hours = config.get("hours", "Business hours")
        services = config.get("services", [])
        service_area = config.get("service_area", "Local area")
        pricing = config.get("pricing", {})
        persona_type = config.get("persona", "friendly").title()
        greeting = config.get("greeting", f"Hello, this is {business_name}. How can I help you?")
        closing = config.get("closing", "Thank you for calling!")
        about = config.get("about_business", "")
        call_flow = config.get("call_flow", {})
        special_instructions = config.get("special_instructions", "")

        # Format services list
        services_text = ", ".join(services) if services else "Various services"

        # Format pricing
        pricing_lines = []
        for service, price in pricing.items():
            service_name = service.replace("_", " ").title()
            pricing_lines.append(f"- {service_name}: {price}")
        pricing_text = "\n".join(pricing_lines) if pricing_lines else "Contact for pricing"

        # Format call flow steps
        flow_steps = call_flow.get("steps", [])
        flow_text = ""
        if flow_steps:
            flow_text = "<mandatory_call_flow>\n"
            flow_text += "<instruction>Follow this flow to guide the conversation naturally. Adapt based on caller needs but ensure you collect required information.</instruction>\n\n"
            for step in flow_steps:
                order = step.get("order", 0)
                name = step.get("name", "")
                action = step.get("action", "")
                collect = step.get("collect", [])
                questions = step.get("questions", [])
                instructions = step.get("instructions", "")

                flow_text += f'<step order="{order}" name="{name}">\n'
                flow_text += f"<action>{action}</action>\n"
                if instructions:
                    flow_text += f"<instructions>{instructions}</instructions>\n"
                if questions:
                    flow_text += "<questions>\n"
                    for q in questions:
                        flow_text += f"  <question>{q}</question>\n"
                    flow_text += "</questions>\n"
                if collect:
                    flow_text += f"<collect>{', '.join(collect)}</collect>\n"
                flow_text += "</step>\n"

            required_data = call_flow.get("required_data", [])
            if required_data:
                flow_text += f"\n<required_data>{', '.join(required_data)}</required_data>\n"

            flow_text += "</mandatory_call_flow>"

        # Build the comprehensive prompt
        prompt = f"""<role>
You are Clark, an AI virtual receptionist for {business_name}. You answer phone calls on behalf of this business. Your goal is to assist callers, gather necessary information, and help with bookings or inquiries.
</role>

<persona type="{persona_type}">
{"You are warm, casual, and welcoming. Use a conversational tone that makes callers feel at ease." if persona_type.lower() == "friendly" else ""}
{"You are calm, compassionate, and understanding. Use a gentle tone and be patient with callers who may be emotional or stressed." if persona_type.lower() == "empathetic" else ""}
{"You are efficient, courteous, and business-like. Be helpful while maintaining a professional demeanor." if persona_type.lower() == "professional" else ""}
</persona>

<voice_only_response_format>
Format all responses as natural spoken words for a voice-only phone conversation.
- Never use text formatting like bullet points, numbered lists, or markdown.
- Never spell out URLs or email addresses character by character.
- Use natural speech patterns and keep responses conversational.
- Keep responses concise - aim for 1-3 sentences per turn.
- Use American English spelling and expressions.
</voice_only_response_format>

<stay_concise>
Be succinct and get straight to the point. Respond directly to the caller with one idea per utterance. Respond in less than three sentences of under twenty words each, unless the caller asks for more detail.
</stay_concise>

<business_information>
<name>{business_name}</name>
<industry>{industry}</industry>
<phone>{phone}</phone>
<timezone>{timezone}</timezone>
<hours>{hours}</hours>
<services>{services_text}</services>
<service_area>{service_area}</service_area>
</business_information>

<pricing>
{pricing_text}
</pricing>

{flow_text}

<greeting>
{greeting}
</greeting>

<closing>
{closing}
</closing>

<about_business>
{about}
</about_business>

{f"<special_instructions>{special_instructions}</special_instructions>" if special_instructions else ""}

<demo_notice>
This is a 3-minute demo call. The caller is testing the Clark AI receptionist system. Be helpful, professional, and demonstrate how Clark handles real calls. If the caller seems to be wrapping up or says goodbye, use the closing message.
</demo_notice>

<behavioral_guidelines>
- Be concise but thorough.
- If you do not know something, say so honestly.
- If the caller seems frustrated, acknowledge their feelings before proceeding.
- End calls professionally with a summary of any actions to be taken.
- Use natural conversational language, not robotic responses.
- NEVER question or express skepticism about information the caller provides.
- Accept all names, locations, and details at face value.
- Your job is to collect information, not to verify it.
</behavioral_guidelines>

<information_collection>
CRITICAL: Do NOT re-ask for information the caller has already provided.
- If the caller mentions a name, relationship, location, or situation, record it and move on.
- When the caller provides multiple pieces of information at once, acknowledge them all and only ask for what's still missing.
- Trust what the caller tells you - do not suggest alternatives or corrections.
- Never say phrases like "I think you might mean...", "Are you sure...?", or "That doesn't sound like..."
</information_collection>"""

        return prompt

    async def fetch_config(self, phone_number: str) -> Optional[BusinessConfig]:
        """Fetch business configuration by phone number.

        Args:
            phone_number: The business's Twilio phone number (To number)

        Returns:
            BusinessConfig if found, None otherwise
        """
        try:
            client = await self._get_client()

            # Normalize phone number
            normalized = phone_number.replace(" ", "").replace("-", "")

            response = await client.get(f"/api/businesses/config/by-phone/{normalized}")

            if response.status_code == 200:
                return self._parse_config_response(response.json())
            else:
                logger.warning(
                    f"Failed to fetch business config for {phone_number}: "
                    f"status={response.status_code}"
                )
                return None

        except Exception as e:
            logger.error(f"Error fetching business config: {e}")
            return None

    async def fetch_config_by_id(self, business_id: str) -> Optional[BusinessConfig]:
        """Fetch business configuration by business ID.

        Used for browser test calls where no phone number is available.

        Args:
            business_id: The business UUID

        Returns:
            BusinessConfig if found, None otherwise
        """
        try:
            client = await self._get_client()

            response = await client.get(f"/api/businesses/config/{business_id}")

            if response.status_code == 200:
                return self._parse_config_response(response.json())
            else:
                logger.warning(
                    f"Failed to fetch business config for id {business_id}: "
                    f"status={response.status_code}"
                )
                return None

        except Exception as e:
            logger.error(f"Error fetching business config by id: {e}")
            return None

    async def handle_event(
        self,
        event: CallEvent,
        emit: Callable[[OperatorOutput], Awaitable[None]],
        ctx: CallContext,
    ) -> None:
        """Handle call events and fetch business config on connection."""

        if not isinstance(event, CallLifecycleEvent):
            return

        if event.kind == "connected":
            # Handle demo calls (from landing page demo)
            # Demo calls load full config from demo_personas.json
            if event.payload.get("is_demo") == "true":
                industry = event.payload.get("industry", "towing")
                logger.info(f"Demo call - loading config for industry: {industry}")

                # Load demo config
                demo_config = self._load_demo_config(industry)
                if not demo_config:
                    logger.error(f"Failed to load demo config for {industry}")
                    return

                # Build comprehensive system prompt
                demo_system_prompt = self._build_demo_system_prompt(demo_config)

                # Emit the system prompt
                await emit(
                    OperatorOutput(
                        kind="system_prompt",
                        payload={
                            "system_prompt": demo_system_prompt,
                            "business_name": demo_config.get("business_name"),
                            "greeting_template": demo_config.get("greeting"),
                            "closing_template": demo_config.get("closing"),
                            "is_demo": True,
                            "style": demo_config.get("persona", "friendly"),
                        },
                    )
                )

                logger.info(f"Demo call configured for {demo_config.get('business_name')}")
                return

            # Check for business_id first (browser test calls pass this directly)
            business_id = event.payload.get("business_id")

            if business_id:
                # Browser test call - fetch by business ID
                logger.info(f"Fetching business config by ID: {business_id}")
                config = await self.fetch_config_by_id(business_id)
            else:
                # Real phone call - fetch by phone number
                to_number = event.payload.get("to_number") or ctx.to_number

                if not to_number:
                    logger.warning(
                        "No 'to' number or business_id in event, cannot fetch business config"
                    )
                    return

                logger.info(f"Fetching business config for {to_number}")
                config = await self.fetch_config(to_number)

            if config:
                self._config = config

                # Log success
                logger.info(
                    f"Loaded config for business: {config.business_name} "
                    f"(industry: {config.industry})"
                )

                # Emit the system prompt for LLM context injection
                # Includes voice_id so TTS can be updated before first greeting
                # Includes style for dynamic emotion selection
                await emit(
                    OperatorOutput(
                        kind="system_prompt",
                        payload={
                            "system_prompt": config.system_prompt,
                            "business_id": config.business_id,
                            "business_name": config.business_name,
                            "greeting_template": config.greeting_template,
                            "closing_template": config.closing_template,
                            "voice_id": config.voice_id,
                            "style": config.style,
                        },
                    )
                )

                # Log the config load
                await emit(
                    OperatorOutput(
                        kind="log",
                        payload={
                            "event": "business_config_loaded",
                            "business_id": config.business_id,
                            "business_name": config.business_name,
                        },
                    )
                )
            else:
                # Determine identifier for error logging
                identifier = business_id or event.payload.get("to_number") or ctx.to_number
                logger.error(f"Failed to load business config for {identifier}")

                # Emit failure event
                await emit(
                    OperatorOutput(
                        kind="log",
                        payload={
                            "event": "business_config_failed",
                            "identifier": identifier,
                        },
                    )
                )

    @property
    def config(self) -> Optional[BusinessConfig]:
        """Get the loaded business configuration."""
        return self._config

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
