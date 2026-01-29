import backoff
import openai
from .pricing import OPENAI_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenAI - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_openai(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query OpenAI model using Chat Completions API."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]

                             
    messages = [{"role": "system", "content": system_msg}] + new_msg_history

                                                                                
    api_kwargs = {}
    if "max_output_tokens" in kwargs:
        api_kwargs["max_tokens"] = kwargs.pop("max_output_tokens")
    if "max_tokens" in kwargs:
        api_kwargs["max_tokens"] = kwargs.pop("max_tokens")
    if "temperature" in kwargs:
        api_kwargs["temperature"] = kwargs.pop("temperature")
    if "seed" in kwargs:
                                                                                       
        api_kwargs["seed"] = kwargs["seed"]
    if "reasoning" in kwargs:
                                                                       
        kwargs.pop("reasoning")

    if output_model is None:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **api_kwargs,
        )
        content = response.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": content})
    else:
                                                
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **api_kwargs,
        )
        content = response.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": content})

                   
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
                                                                  
    model_pricing = OPENAI_MODELS.get(model, {"input_price": 0.0, "output_price": 0.0})
    input_cost = model_pricing["input_price"] * input_tokens
    output_cost = model_pricing["output_price"] * output_tokens

    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",
        model_posteriors=model_posteriors,
    )
    return result
