import logging
from typing import Dict, List, Union, Optional
import re
import json
import multiprocessing as mp
import asyncio
from pydantic import BaseModel
import time
import os
from .query import sample_model_kwargs, query
from .models import QueryResult
from .dynamic_sampling import BanditBase, FixedSampler

MAX_RETRIES = 3

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        model_names: Union[List[str], str] = "gpt-4o-2024-05-13",
        model_selection: Optional[BanditBase] = None,
        temperatures: Union[float, List[float]] = 0.75,
        max_tokens: Union[int, List[int]] = 4096,
        reasoning_efforts: Union[str, List[str]] = "auto",
        model_sample_probs: Optional[List[float]] = None,
        output_model: Optional[BaseModel] = None,
        base_url: Optional[str] = None,
        verbose: bool = True,
        max_total_calls: Optional[int] = None,
    ):
        self.temperatures = temperatures
        self.max_tokens = max_tokens
        if isinstance(model_names, str):
            model_names = [model_names]
        self.model_names = model_names
        if not isinstance(model_selection, BanditBase):
            assert model_selection is None
            model_selection = FixedSampler(
                n_arms=len(model_names),
                prior_probs=model_sample_probs,
            )
        self.llm_selection = model_selection
        self.reasoning_efforts = reasoning_efforts
        self.model_sample_probs = model_sample_probs
        self.output_model = output_model
        self.structured_output = output_model is not None
        self.base_url = str(base_url).strip() if base_url else None
        self.verbose = verbose
        self.total_calls: int = 0
        self.max_total_calls: Optional[int] = (
            int(max_total_calls) if max_total_calls is not None else None
        )

    def _budget_exhausted(self) -> bool:
        return self.max_total_calls is not None and self.total_calls >= self.max_total_calls

    def batch_query(
        self,
        num_samples: int,
        msg: Union[str, List[str]],
        system_msg: Union[str, List[str]],
        msg_history: Union[List[Dict], List[List[Dict]]] = [],
        llm_kwargs: List[Dict] = [],
    ) -> List[QueryResult]:
        """Batch query the LLM with the given message and system message.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
        """
        if self.max_total_calls is not None:
            remaining = int(self.max_total_calls) - int(self.total_calls)
            if remaining <= 0:
                logger.warning(
                    "LLM budget exhausted: total_calls=%s max_total_calls=%s (batch_query)",
                    self.total_calls,
                    self.max_total_calls,
                )
                return []
            if int(num_samples) > remaining:
                logger.warning(
                    "LLM budget nearly exhausted: truncating batch_query from %s to %s "
                    "(total_calls=%s max_total_calls=%s)",
                    num_samples,
                    remaining,
                    self.total_calls,
                    self.max_total_calls,
                )
                num_samples = int(remaining)

        self.total_calls += int(num_samples)
                                                               
        if isinstance(msg, str):
            msg = [msg] * num_samples
        if isinstance(system_msg, str):
            system_msg = [system_msg] * num_samples
        if len(msg_history) == 0:
            msg_history = [[]] * num_samples
        elif isinstance(msg_history[0], dict):
            msg_history = [msg_history] * num_samples

        if isinstance(msg, list) and len(msg) > num_samples:
            msg = msg[:num_samples]
        if isinstance(system_msg, list) and len(system_msg) > num_samples:
            system_msg = system_msg[:num_samples]
        if isinstance(msg_history, list) and len(msg_history) > num_samples:
            msg_history = msg_history[:num_samples]
        if isinstance(llm_kwargs, list) and len(llm_kwargs) > num_samples:
            llm_kwargs = llm_kwargs[:num_samples]

                                          
        num_processes = min(num_samples, mp.cpu_count())
        with mp.Pool(processes=num_processes) as pool:
                                                   
            async_results = []
            for i in range(len(msg)):
                async_results.append(
                    pool.apply_async(
                        query_fn,
                        args=(
                            i,
                            msg[i],
                            system_msg[i],
                            msg_history[i],
                            llm_kwargs[i],
                            num_samples,
                            self.output_model,
                            self.base_url,
                            self.verbose,
                        ),
                    )
                )

                                                        
            results = []
            for async_result in async_results:
                try:
                    idx, result = async_result.get()
                    results.append((idx, result))
                except Exception as e:
                    logger.error(f"Error in batch query: {str(e)}")

                                                        
            results.sort(key=lambda x: x[0])
            final_results = [r[1] for r in results if r[1] is not None]

                                    
            if self.verbose:
                total_cost = sum(
                    r.cost
                    for r in final_results
                    if hasattr(r, "cost") and r.cost is not None
                )
                formatted_costs = [
                    f"{r.cost:.4f}"
                    for r in final_results
                    if hasattr(r, "cost") and r.cost is not None
                ]
                logger.info(f"==> SAMPLING: Individual API costs: {formatted_costs}")
                logger.info(f"==> SAMPLING: Total API costs: ${total_cost:.4f}")
            return final_results

    def batch_kwargs_query(
        self,
        num_samples: int,
        msg: Union[str, List[str]],
        system_msg: Union[str, List[str]],
        msg_history: Union[List[Dict], List[List[Dict]]] = [],
    ) -> List[QueryResult]:
        """Batch query the LLM with the given message and system message.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
        """
        if self.max_total_calls is not None:
            remaining = int(self.max_total_calls) - int(self.total_calls)
            if remaining <= 0:
                logger.warning(
                    "LLM budget exhausted: total_calls=%s max_total_calls=%s (batch_kwargs_query)",
                    self.total_calls,
                    self.max_total_calls,
                )
                return []
            if int(num_samples) > remaining:
                logger.warning(
                    "LLM budget nearly exhausted: truncating batch_kwargs_query from %s to %s "
                    "(total_calls=%s max_total_calls=%s)",
                    num_samples,
                    remaining,
                    self.total_calls,
                    self.max_total_calls,
                )
                num_samples = int(remaining)

        self.total_calls += int(num_samples)
                                                               
        if isinstance(msg, str):
            msg = [msg] * num_samples
        if isinstance(system_msg, str):
            system_msg = [system_msg] * num_samples
        if len(msg_history) == 0:
            msg_history = [[]] * num_samples
        elif isinstance(msg_history[0], dict):
            msg_history = [msg_history] * num_samples

        if isinstance(msg, list) and len(msg) > num_samples:
            msg = msg[:num_samples]
        if isinstance(system_msg, list) and len(system_msg) > num_samples:
            system_msg = system_msg[:num_samples]
        if isinstance(msg_history, list) and len(msg_history) > num_samples:
            msg_history = msg_history[:num_samples]

                                          
        num_processes = min(num_samples, mp.cpu_count())
        with mp.Pool(processes=num_processes) as pool:
                                                   
            async_results = []
            posterior = self.llm_selection.posterior(samples=num_samples)
            if self.verbose:
                lines = [f"==> SAMPLING {num_samples} SAMPLES:"]
                for name, prob in zip(self.model_names, posterior):
                    lines.append(f"  {name:<30} {prob:>8.4f}")
                logger.info("\n".join(lines))
            for i in range(len(msg)):
                async_results.append(
                    pool.apply_async(
                        sample_kwargs_query_fn,
                        args=(
                            i,
                            msg[i],
                            system_msg[i],
                            msg_history[i],
                            self.model_names,
                            self.temperatures,
                            self.max_tokens,
                            self.reasoning_efforts,
                            posterior,
                            self.output_model,
                            num_samples,
                            self.base_url,
                            self.verbose,
                        ),
                    )
                )

                                                        
            results = []
            for async_result in async_results:
                try:
                    idx, result = async_result.get()
                    results.append((idx, result))
                except Exception as e:
                    logger.error(f"Error in batch query: {str(e)}")

                                                        
            results.sort(key=lambda x: x[0])
            final_results = [r[1] for r in results if r[1] is not None]

                                    
            if self.verbose:
                total_cost = sum(
                    r.cost
                    for r in final_results
                    if hasattr(r, "cost") and r.cost is not None
                )
                formatted_costs = [
                    f"{r.cost:.4f}"
                    for r in final_results
                    if hasattr(r, "cost") and r.cost is not None
                ]
                logger.info(f"==> SAMPLING: Individual API costs: {formatted_costs}")
                logger.info(f"==> SAMPLING: Total API costs: ${total_cost:.4f}")
            return final_results

    def get_kwargs(self):
        posterior = self.llm_selection.posterior()
        if self.verbose:
            lines = ["==> SAMPLING:"]
            for name, prob in zip(self.model_names, posterior):
                lines.append(f"  {name:<30} {prob:>8.4f}")
            logger.info("\n".join(lines))
        return sample_model_kwargs(
            model_names=self.model_names,
            temperatures=self.temperatures,
            max_tokens=self.max_tokens,
            reasoning_efforts=self.reasoning_efforts,
            model_sample_probs=posterior,
        )

    def query(
        self,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        llm_kwargs: Optional[Dict] = None,
    ) -> Optional[QueryResult]:
        """Execute a single query to the LLM.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
            msg_history (List[Dict], optional): Message history. Defaults to [].
            llm_kwargs (Dict, optional): Additional LLM parameters.
                Defaults to {}.

        Returns:
            QueryResult: The result of the query.
        """
        if llm_kwargs is None:
            llm_kwargs = sample_model_kwargs(
                model_names=self.model_names,
                temperatures=self.temperatures,
                max_tokens=self.max_tokens,
                reasoning_efforts=self.reasoning_efforts,
                model_sample_probs=self.model_sample_probs,
            )
        if self.verbose:
            logger.info(f"==> QUERYING: {list(llm_kwargs.values())}")

                                                                      
        posterior = self.llm_selection.posterior()
        model_posteriors = dict(zip(self.model_names, posterior))
        model_posteriors = {k: float(v) for k, v in model_posteriors.items()}
        try_count = 0
        while try_count < MAX_RETRIES:
            if self._budget_exhausted():
                logger.warning(
                    "LLM budget exhausted: total_calls=%s max_total_calls=%s (query)",
                    self.total_calls,
                    self.max_total_calls,
                )
                return None
            try:
                self.total_calls += 1
                kwargs_to_send = llm_kwargs
                                                                                    
                                                                             
                                                                                         
                seed_base = os.environ.get("AUTOFORMAL_SEED")
                if seed_base and "seed" not in (llm_kwargs or {}):
                    try:
                        seed_val = int(seed_base) + int(self.total_calls)
                        kwargs_to_send = dict(llm_kwargs or {})
                        kwargs_to_send["seed"] = seed_val
                    except Exception:
                        kwargs_to_send = llm_kwargs
                result = query(
                    msg=msg,
                    system_msg=system_msg,
                    msg_history=msg_history,
                    output_model=self.output_model,
                    model_posteriors=model_posteriors,
                    base_url=self.base_url,
                    **(kwargs_to_send or {}),
                )
                if self.verbose and hasattr(result, "cost") and result.cost is not None:
                    logger.info(f"==> QUERY: API cost: ${result.cost:.4f}")
                return result
            except Exception as e:
                logger.error(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                try_count += 1
        return None


class AsyncLLMClient:
    def __init__(
        self,
        model_names: Union[List[str], str] = "gpt-4o-2024-05-13",
        model_selection: Optional[BanditBase] = None,
        temperatures: Union[float, List[float]] = 0.75,
        max_tokens: Union[int, List[int]] = 4096,
        reasoning_efforts: Union[str, List[str]] = "auto",
        model_sample_probs: Optional[List[float]] = None,
        output_model: Optional[BaseModel] = None,
        verbose: bool = True,
    ):
        self.temperatures = temperatures
        self.max_tokens = max_tokens
        if isinstance(model_names, str):
            model_names = [model_names]
        self.model_names = model_names
        if not isinstance(model_selection, BanditBase):
            assert model_selection is None
            model_selection = FixedSampler(
                n_arms=len(model_names),
                prior_probs=model_sample_probs,
            )
        self.llm_selection = model_selection
        self.reasoning_efforts = reasoning_efforts
        self.model_sample_probs = model_sample_probs
        self.output_model = output_model
        self.structured_output = output_model is not None
        self.verbose = verbose

    async def batch_query(
        self,
        num_samples: int,
        msg: Union[str, List[str]],
        system_msg: Union[str, List[str]],
        msg_history: Union[List[Dict], List[List[Dict]]] = [],
        llm_kwargs: List[Dict] = [],
    ) -> List[QueryResult]:
        """Batch query the LLM with the given message and system message asynchronously.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
        """
                                                               
        if isinstance(msg, str):
            msg = [msg] * num_samples
        if isinstance(system_msg, str):
            system_msg = [system_msg] * num_samples
        if len(msg_history) == 0:
            msg_history = [[]] * num_samples
        elif isinstance(msg_history[0], dict):
            msg_history = [msg_history] * num_samples

                            
        tasks = []
        for i in range(len(msg)):
            tasks.append(
                self._query_async_with_retry(
                    i,
                    msg[i],
                    system_msg[i],
                    msg_history[i],
                    llm_kwargs[i],
                    num_samples,
                )
            )

                                        
        results = await asyncio.gather(*tasks, return_exceptions=True)

                                                   
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.info(f"Error in batch query task {i}: {str(result)}")
            elif result is not None and len(result) > 1 and result[1] is not None:
                final_results.append(result[1])

                                
        if self.verbose:
            total_cost = sum(
                r.cost
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            )
            formatted_costs = [
                f"{r.cost:.4f}"
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            ]
            logger.info(f"==> SAMPLING: Individual API costs: {formatted_costs}")
            logger.info(f"==> SAMPLING: Total API costs: ${total_cost:.4f}")
        return final_results

    async def batch_kwargs_query(
        self,
        num_samples: int,
        msg: Union[str, List[str]],
        system_msg: Union[str, List[str]],
        msg_history: Union[List[Dict], List[List[Dict]]] = [],
    ) -> List[QueryResult]:
        """Batch query the LLM with the given message and system message asynchronously.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
        """
                                                               
        if isinstance(msg, str):
            msg = [msg] * num_samples
        if isinstance(system_msg, str):
            system_msg = [system_msg] * num_samples
        if len(msg_history) == 0:
            msg_history = [[]] * num_samples
        elif isinstance(msg_history[0], dict):
            msg_history = [msg_history] * num_samples

                                     
        posterior = self.llm_selection.posterior(samples=num_samples)
        if self.verbose:
            lines = [f"==> SAMPLING {num_samples} SAMPLES:"]
            for name, prob in zip(self.model_names, posterior):
                lines.append(f"  {name:<30} {prob:>8.4f}")
            logger.info("\n".join(lines))

                            
        tasks = []
        for i in range(len(msg)):
            tasks.append(
                self._sample_kwargs_query_async_with_retry(
                    i,
                    msg[i],
                    system_msg[i],
                    msg_history[i],
                    posterior,
                    num_samples,
                )
            )

                                        
        results = await asyncio.gather(*tasks, return_exceptions=True)

                                                   
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.info(f"Error in batch query task {i}: {str(result)}")
            elif result is not None and len(result) > 1 and result[1] is not None:
                final_results.append(result[1])

                                
        if self.verbose:
            total_cost = sum(
                r.cost
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            )
            formatted_costs = [
                f"{r.cost:.4f}"
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            ]
            logger.info(f"==> SAMPLING: Individual API costs: {formatted_costs}")
            logger.info(f"==> SAMPLING: Total API costs: ${total_cost:.4f}")
        return final_results

    def get_kwargs(self):
        posterior = self.llm_selection.posterior()
        if self.verbose:
            lines = ["==> SAMPLING:"]
            for name, prob in zip(self.model_names, posterior):
                lines.append(f"  {name:<30} {prob:>8.4f}")
            logger.info("\n".join(lines))
        return sample_model_kwargs(
            model_names=self.model_names,
            temperatures=self.temperatures,
            max_tokens=self.max_tokens,
            reasoning_efforts=self.reasoning_efforts,
            model_sample_probs=posterior,
        )

    async def query(
        self,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        llm_kwargs: Optional[Dict] = None,
    ) -> Optional[QueryResult]:
        """Execute a single query to the LLM asynchronously.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
            msg_history (List[Dict], optional): Message history. Defaults to [].
            llm_kwargs (Dict, optional): Additional LLM parameters.
                Defaults to {}.

        Returns:
            QueryResult: The result of the query.
        """
        if llm_kwargs is None:
            llm_kwargs = sample_model_kwargs(
                model_names=self.model_names,
                temperatures=self.temperatures,
                max_tokens=self.max_tokens,
                reasoning_efforts=self.reasoning_efforts,
                model_sample_probs=self.model_sample_probs,
            )
        if self.verbose:
            logger.info(f"==> QUERYING: {list(llm_kwargs.values())}")

                                                                      
        posterior = self.llm_selection.posterior()
        model_posteriors = dict(zip(self.model_names, posterior))
        model_posteriors = {k: float(v) for k, v in model_posteriors.items()}

        try_count = 0
        while try_count < MAX_RETRIES:
            try:
                result = await query_async(
                    msg=msg,
                    system_msg=system_msg,
                    msg_history=msg_history,
                    output_model=self.output_model,
                    model_posteriors=model_posteriors,
                    **llm_kwargs,
                )
                if self.verbose and hasattr(result, "cost") and result.cost is not None:
                    logger.info(f"==> QUERY: API cost: ${result.cost:.4f}")
                return result
            except Exception as e:
                logger.info(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                try_count += 1
                if try_count < MAX_RETRIES:
                    await asyncio.sleep(1)                             
        return None

    async def _query_async_with_retry(
        self,
        idx: int,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        kwargs: Dict = {},
        total_samples: int = 1,
    ) -> tuple[int, Optional[QueryResult]]:
        if self.verbose:
            logger.info(
                f"==> SAMPLING: {idx + 1}/{total_samples} {list(kwargs.values())}"
            )

        try_count = 0
        while try_count < MAX_RETRIES:
            try:
                result = await query_async(
                    msg=msg,
                    system_msg=system_msg,
                    msg_history=msg_history,
                    output_model=self.output_model,
                    **kwargs,
                )
                return idx, result
            except Exception as e:
                logger.info(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                try_count += 1
                if try_count == MAX_RETRIES:
                    return idx, None
                await asyncio.sleep(1)                             
        return idx, None

    async def _sample_kwargs_query_async_with_retry(
        self,
        idx: int,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        model_sample_probs: Optional[List[float]] = None,
        total_samples: int = 1,
    ) -> tuple[int, Optional[QueryResult]]:
        kwargs = sample_model_kwargs(
            model_names=self.model_names,
            temperatures=self.temperatures,
            max_tokens=self.max_tokens,
            reasoning_efforts=self.reasoning_efforts,
            model_sample_probs=model_sample_probs,
        )

                                                                              
        model_posteriors = None
        if model_sample_probs is not None and isinstance(self.model_names, list):
            model_posteriors = dict(zip(self.model_names, model_sample_probs))
            model_posteriors = {k: float(v) for k, v in model_posteriors.items()}

        if self.verbose:
            logger.info(
                f"==> SAMPLING: {idx + 1}/{total_samples} {list(kwargs.values())}"
            )

        try_count = 0
        while try_count < MAX_RETRIES:
            try:
                result = await query_async(
                    msg=msg,
                    system_msg=system_msg,
                    msg_history=msg_history,
                    output_model=self.output_model,
                    model_posteriors=model_posteriors,
                    **kwargs,
                )
                return idx, result
            except Exception as e:
                logger.info(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                try_count += 1
                if try_count == MAX_RETRIES:
                    return idx, None
                await asyncio.sleep(1)                             
        return idx, None


class AsyncLLMClient:
    def __init__(
        self,
        model_names: Union[List[str], str] = "gpt-4o-2024-05-13",
        model_selection: Optional[BanditBase] = None,
        temperatures: Union[float, List[float]] = 0.75,
        max_tokens: Union[int, List[int]] = 4096,
        reasoning_efforts: Union[str, List[str]] = "auto",
        model_sample_probs: Optional[List[float]] = None,
        output_model: Optional[BaseModel] = None,
        verbose: bool = True,
    ):
        self.temperatures = temperatures
        self.max_tokens = max_tokens
        if isinstance(model_names, str):
            model_names = [model_names]
        self.model_names = model_names
        if not isinstance(model_selection, BanditBase):
            assert model_selection is None
            model_selection = FixedSampler(
                n_arms=len(model_names),
                prior_probs=model_sample_probs,
            )
        self.llm_selection = model_selection
        self.reasoning_efforts = reasoning_efforts
        self.model_sample_probs = model_sample_probs
        self.output_model = output_model
        self.structured_output = output_model is not None
        self.verbose = verbose

    async def batch_query(
        self,
        num_samples: int,
        msg: Union[str, List[str]],
        system_msg: Union[str, List[str]],
        msg_history: Union[List[Dict], List[List[Dict]]] = [],
        llm_kwargs: List[Dict] = [],
    ) -> List[QueryResult]:
        """Batch query the LLM with the given message and system message asynchronously.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
        """
                                                               
        if isinstance(msg, str):
            msg = [msg] * num_samples
        if isinstance(system_msg, str):
            system_msg = [system_msg] * num_samples
        if len(msg_history) == 0:
            msg_history = [[]] * num_samples
        elif isinstance(msg_history[0], dict):
            msg_history = [msg_history] * num_samples

                            
        tasks = []
        for i in range(len(msg)):
            tasks.append(
                self._query_async_with_retry(
                    i,
                    msg[i],
                    system_msg[i],
                    msg_history[i],
                    llm_kwargs[i],
                    num_samples,
                )
            )

                                        
        results = await asyncio.gather(*tasks, return_exceptions=True)

                                                   
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.info(f"Error in batch query task {i}: {str(result)}")
            elif result is not None and len(result) > 1 and result[1] is not None:
                final_results.append(result[1])

                                
        if self.verbose:
            total_cost = sum(
                r.cost
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            )
            formatted_costs = [
                f"{r.cost:.4f}"
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            ]
            logger.info(f"==> SAMPLING: Individual API costs: {formatted_costs}")
            logger.info(f"==> SAMPLING: Total API costs: ${total_cost:.4f}")
        return final_results

    async def batch_kwargs_query(
        self,
        num_samples: int,
        msg: Union[str, List[str]],
        system_msg: Union[str, List[str]],
        msg_history: Union[List[Dict], List[List[Dict]]] = [],
    ) -> List[QueryResult]:
        """Batch query the LLM with the given message and system message asynchronously.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
        """
                                                               
        if isinstance(msg, str):
            msg = [msg] * num_samples
        if isinstance(system_msg, str):
            system_msg = [system_msg] * num_samples
        if len(msg_history) == 0:
            msg_history = [[]] * num_samples
        elif isinstance(msg_history[0], dict):
            msg_history = [msg_history] * num_samples

                                     
        posterior = self.llm_selection.posterior(samples=num_samples)
        if self.verbose:
            lines = [f"==> SAMPLING {num_samples} SAMPLES:"]
            for name, prob in zip(self.model_names, posterior):
                lines.append(f"  {name:<30} {prob:>8.4f}")
            logger.info("\n".join(lines))

                            
        tasks = []
        for i in range(len(msg)):
            tasks.append(
                self._sample_kwargs_query_async_with_retry(
                    i,
                    msg[i],
                    system_msg[i],
                    msg_history[i],
                    posterior,
                    num_samples,
                )
            )

                                        
        results = await asyncio.gather(*tasks, return_exceptions=True)

                                                   
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.info(f"Error in batch query task {i}: {str(result)}")
            elif result is not None and len(result) > 1 and result[1] is not None:
                final_results.append(result[1])

                                
        if self.verbose:
            total_cost = sum(
                r.cost
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            )
            formatted_costs = [
                f"{r.cost:.4f}"
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            ]
            logger.info(f"==> SAMPLING: Individual API costs: {formatted_costs}")
            logger.info(f"==> SAMPLING: Total API costs: ${total_cost:.4f}")
        return final_results

    def get_kwargs(self):
        posterior = self.llm_selection.posterior()
        if self.verbose:
            lines = ["==> SAMPLING:"]
            for name, prob in zip(self.model_names, posterior):
                lines.append(f"  {name:<30} {prob:>8.4f}")
            logger.info("\n".join(lines))
        return sample_model_kwargs(
            model_names=self.model_names,
            temperatures=self.temperatures,
            max_tokens=self.max_tokens,
            reasoning_efforts=self.reasoning_efforts,
            model_sample_probs=posterior,
        )

    async def query(
        self,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        llm_kwargs: Optional[Dict] = None,
    ) -> Optional[QueryResult]:
        """Execute a single query to the LLM asynchronously.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
            msg_history (List[Dict], optional): Message history. Defaults to [].
            llm_kwargs (Dict, optional): Additional LLM parameters.
                Defaults to {}.

        Returns:
            QueryResult: The result of the query.
        """
        if llm_kwargs is None:
            llm_kwargs = sample_model_kwargs(
                model_names=self.model_names,
                temperatures=self.temperatures,
                max_tokens=self.max_tokens,
                reasoning_efforts=self.reasoning_efforts,
                model_sample_probs=self.model_sample_probs,
            )
        if self.verbose:
            logger.info(f"==> QUERYING: {list(llm_kwargs.values())}")

                                                                      
        posterior = self.llm_selection.posterior()
        model_posteriors = dict(zip(self.model_names, posterior))
        model_posteriors = {k: float(v) for k, v in model_posteriors.items()}

        try_count = 0
        while try_count < MAX_RETRIES:
            try:
                result = await query_async(
                    msg=msg,
                    system_msg=system_msg,
                    msg_history=msg_history,
                    output_model=self.output_model,
                    model_posteriors=model_posteriors,
                    **llm_kwargs,
                )
                if self.verbose and hasattr(result, "cost") and result.cost is not None:
                    logger.info(f"==> QUERY: API cost: ${result.cost:.4f}")
                return result
            except Exception as e:
                logger.info(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                try_count += 1
                if try_count < MAX_RETRIES:
                    await asyncio.sleep(1)                             
        return None

    async def _query_async_with_retry(
        self,
        idx: int,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        kwargs: Dict = {},
        total_samples: int = 1,
    ) -> tuple[int, Optional[QueryResult]]:
        if self.verbose:
            logger.info(
                f"==> SAMPLING: {idx + 1}/{total_samples} {list(kwargs.values())}"
            )

        try_count = 0
        while try_count < MAX_RETRIES:
            try:
                result = await query_async(
                    msg=msg,
                    system_msg=system_msg,
                    msg_history=msg_history,
                    output_model=self.output_model,
                    **kwargs,
                )
                return idx, result
            except Exception as e:
                logger.info(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                try_count += 1
                if try_count == MAX_RETRIES:
                    return idx, None
                await asyncio.sleep(1)                             
        return idx, None

    async def _sample_kwargs_query_async_with_retry(
        self,
        idx: int,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        model_sample_probs: Optional[List[float]] = None,
        total_samples: int = 1,
    ) -> tuple[int, Optional[QueryResult]]:
        kwargs = sample_model_kwargs(
            model_names=self.model_names,
            temperatures=self.temperatures,
            max_tokens=self.max_tokens,
            reasoning_efforts=self.reasoning_efforts,
            model_sample_probs=model_sample_probs,
        )

                                                                              
        model_posteriors = None
        if model_sample_probs is not None and isinstance(self.model_names, list):
            model_posteriors = dict(zip(self.model_names, model_sample_probs))
            model_posteriors = {k: float(v) for k, v in model_posteriors.items()}

        if self.verbose:
            logger.info(
                f"==> SAMPLING: {idx + 1}/{total_samples} {list(kwargs.values())}"
            )

        try_count = 0
        while try_count < MAX_RETRIES:
            try:
                result = await query_async(
                    msg=msg,
                    system_msg=system_msg,
                    msg_history=msg_history,
                    output_model=self.output_model,
                    model_posteriors=model_posteriors,
                    **kwargs,
                )
                return idx, result
            except Exception as e:
                logger.info(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                try_count += 1
                if try_count == MAX_RETRIES:
                    return idx, None
                await asyncio.sleep(1)                             
        return idx, None


class AsyncLLMClient:
    def __init__(
        self,
        model_names: Union[List[str], str] = "gpt-4o-2024-05-13",
        model_selection: Optional[BanditBase] = None,
        temperatures: Union[float, List[float]] = 0.75,
        max_tokens: Union[int, List[int]] = 4096,
        reasoning_efforts: Union[str, List[str]] = "auto",
        model_sample_probs: Optional[List[float]] = None,
        output_model: Optional[BaseModel] = None,
        verbose: bool = True,
    ):
        self.temperatures = temperatures
        self.max_tokens = max_tokens
        if isinstance(model_names, str):
            model_names = [model_names]
        self.model_names = model_names
        if not isinstance(model_selection, BanditBase):
            assert model_selection is None
            model_selection = FixedSampler(
                n_arms=len(model_names),
                prior_probs=model_sample_probs,
            )
        self.llm_selection = model_selection
        self.reasoning_efforts = reasoning_efforts
        self.model_sample_probs = model_sample_probs
        self.output_model = output_model
        self.structured_output = output_model is not None
        self.verbose = verbose

    async def batch_query(
        self,
        num_samples: int,
        msg: Union[str, List[str]],
        system_msg: Union[str, List[str]],
        msg_history: Union[List[Dict], List[List[Dict]]] = [],
        llm_kwargs: List[Dict] = [],
    ) -> List[QueryResult]:
        """Batch query the LLM with the given message and system message asynchronously.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
        """
                                                               
        if isinstance(msg, str):
            msg = [msg] * num_samples
        if isinstance(system_msg, str):
            system_msg = [system_msg] * num_samples
        if len(msg_history) == 0:
            msg_history = [[]] * num_samples
        elif isinstance(msg_history[0], dict):
            msg_history = [msg_history] * num_samples

                            
        tasks = []
        for i in range(len(msg)):
            tasks.append(
                self._query_async_with_retry(
                    i,
                    msg[i],
                    system_msg[i],
                    msg_history[i],
                    llm_kwargs[i],
                    num_samples,
                )
            )

                                        
        results = await asyncio.gather(*tasks, return_exceptions=True)

                                                   
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.info(f"Error in batch query task {i}: {str(result)}")
            elif result is not None and len(result) > 1 and result[1] is not None:
                final_results.append(result[1])

                                
        if self.verbose:
            total_cost = sum(
                r.cost
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            )
            formatted_costs = [
                f"{r.cost:.4f}"
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            ]
            logger.info(f"==> SAMPLING: Individual API costs: {formatted_costs}")
            logger.info(f"==> SAMPLING: Total API costs: ${total_cost:.4f}")
        return final_results

    async def batch_kwargs_query(
        self,
        num_samples: int,
        msg: Union[str, List[str]],
        system_msg: Union[str, List[str]],
        msg_history: Union[List[Dict], List[List[Dict]]] = [],
    ) -> List[QueryResult]:
        """Batch query the LLM with the given message and system message asynchronously.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
        """
                                                               
        if isinstance(msg, str):
            msg = [msg] * num_samples
        if isinstance(system_msg, str):
            system_msg = [system_msg] * num_samples
        if len(msg_history) == 0:
            msg_history = [[]] * num_samples
        elif isinstance(msg_history[0], dict):
            msg_history = [msg_history] * num_samples

                                     
        posterior = self.llm_selection.posterior(samples=num_samples)
        if self.verbose:
            lines = [f"==> SAMPLING {num_samples} SAMPLES:"]
            for name, prob in zip(self.model_names, posterior):
                lines.append(f"  {name:<30} {prob:>8.4f}")
            logger.info("\n".join(lines))

                            
        tasks = []
        for i in range(len(msg)):
            tasks.append(
                self._sample_kwargs_query_async_with_retry(
                    i,
                    msg[i],
                    system_msg[i],
                    msg_history[i],
                    posterior,
                    num_samples,
                )
            )

                                        
        results = await asyncio.gather(*tasks, return_exceptions=True)

                                                   
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.info(f"Error in batch query task {i}: {str(result)}")
            elif result is not None and len(result) > 1 and result[1] is not None:
                final_results.append(result[1])

                                
        if self.verbose:
            total_cost = sum(
                r.cost
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            )
            formatted_costs = [
                f"{r.cost:.4f}"
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            ]
            logger.info(f"==> SAMPLING: Individual API costs: {formatted_costs}")
            logger.info(f"==> SAMPLING: Total API costs: ${total_cost:.4f}")
        return final_results

    def get_kwargs(self):
        posterior = self.llm_selection.posterior()
        if self.verbose:
            lines = ["==> SAMPLING:"]
            for name, prob in zip(self.model_names, posterior):
                lines.append(f"  {name:<30} {prob:>8.4f}")
            logger.info("\n".join(lines))
        return sample_model_kwargs(
            model_names=self.model_names,
            temperatures=self.temperatures,
            max_tokens=self.max_tokens,
            reasoning_efforts=self.reasoning_efforts,
            model_sample_probs=posterior,
        )

    async def query(
        self,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        llm_kwargs: Optional[Dict] = None,
    ) -> Optional[QueryResult]:
        """Execute a single query to the LLM asynchronously.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
            msg_history (List[Dict], optional): Message history. Defaults to [].
            llm_kwargs (Dict, optional): Additional LLM parameters.
                Defaults to {}.

        Returns:
            QueryResult: The result of the query.
        """
        if llm_kwargs is None:
            llm_kwargs = sample_model_kwargs(
                model_names=self.model_names,
                temperatures=self.temperatures,
                max_tokens=self.max_tokens,
                reasoning_efforts=self.reasoning_efforts,
                model_sample_probs=self.model_sample_probs,
            )
        if self.verbose:
            logger.info(f"==> QUERYING: {list(llm_kwargs.values())}")

                                                                      
        posterior = self.llm_selection.posterior()
        model_posteriors = dict(zip(self.model_names, posterior))
        model_posteriors = {k: float(v) for k, v in model_posteriors.items()}

        try_count = 0
        while try_count < MAX_RETRIES:
            try:
                result = await query_async(
                    msg=msg,
                    system_msg=system_msg,
                    msg_history=msg_history,
                    output_model=self.output_model,
                    model_posteriors=model_posteriors,
                    **llm_kwargs,
                )
                if self.verbose and hasattr(result, "cost") and result.cost is not None:
                    logger.info(f"==> QUERY: API cost: ${result.cost:.4f}")
                return result
            except Exception as e:
                logger.info(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                try_count += 1
                if try_count < MAX_RETRIES:
                    await asyncio.sleep(1)                             
        return None

    async def _query_async_with_retry(
        self,
        idx: int,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        kwargs: Dict = {},
        total_samples: int = 1,
    ) -> tuple[int, Optional[QueryResult]]:
        if self.verbose:
            logger.info(
                f"==> SAMPLING: {idx + 1}/{total_samples} {list(kwargs.values())}"
            )

        try_count = 0
        while try_count < MAX_RETRIES:
            try:
                result = await query_async(
                    msg=msg,
                    system_msg=system_msg,
                    msg_history=msg_history,
                    output_model=self.output_model,
                    **kwargs,
                )
                return idx, result
            except Exception as e:
                logger.info(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                try_count += 1
                if try_count == MAX_RETRIES:
                    return idx, None
                await asyncio.sleep(1)                             
        return idx, None

    async def _sample_kwargs_query_async_with_retry(
        self,
        idx: int,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        model_sample_probs: Optional[List[float]] = None,
        total_samples: int = 1,
    ) -> tuple[int, Optional[QueryResult]]:
        kwargs = sample_model_kwargs(
            model_names=self.model_names,
            temperatures=self.temperatures,
            max_tokens=self.max_tokens,
            reasoning_efforts=self.reasoning_efforts,
            model_sample_probs=model_sample_probs,
        )

                                                                              
        model_posteriors = None
        if model_sample_probs is not None and isinstance(self.model_names, list):
            model_posteriors = dict(zip(self.model_names, model_sample_probs))
            model_posteriors = {k: float(v) for k, v in model_posteriors.items()}

        if self.verbose:
            logger.info(
                f"==> SAMPLING: {idx + 1}/{total_samples} {list(kwargs.values())}"
            )

        try_count = 0
        while try_count < MAX_RETRIES:
            try:
                result = await query_async(
                    msg=msg,
                    system_msg=system_msg,
                    msg_history=msg_history,
                    output_model=self.output_model,
                    model_posteriors=model_posteriors,
                    **kwargs,
                )
                return idx, result
            except Exception as e:
                logger.info(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                try_count += 1
                if try_count == MAX_RETRIES:
                    return idx, None
                await asyncio.sleep(1)                             
        return idx, None


def query_fn(
    idx: int,
    msg: str,
    system_msg: str,
    msg_history: List[Dict] = [],
    kwargs: Dict = {},
    total_samples: int = 1,
    output_model: Optional[BaseModel] = None,
    base_url: Optional[str] = None,
    verbose: bool = False,
) -> tuple[int, Optional[QueryResult]]:
    if verbose:
        logger.info(f"==> SAMPLING: {idx + 1}/{total_samples} {list(kwargs.values())}")
    try_count = 0
    while try_count < MAX_RETRIES:
        try:
            result = query(
                msg=msg,
                system_msg=system_msg,
                msg_history=msg_history,
                output_model=output_model,
                base_url=base_url,
                **kwargs,
            )
            return idx, result
        except Exception as e:
            logger.error(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
            try_count += 1
            if try_count == MAX_RETRIES:
                                                      
                return idx, None
                                               
            time.sleep(1)
    return idx, None


def sample_kwargs_query_fn(
    idx: int,
    msg: str,
    system_msg: str,
    msg_history: List[Dict] = [],
    model_names: Union[List[str], str] = "gpt-4o-2024-05-13",
    temperatures: Union[float, List[float]] = 0.75,
    max_tokens: Union[int, List[int]] = 4096,
    reasoning_efforts: Union[str, List[str]] = "high",
    model_sample_probs: Optional[List[float]] = None,
    output_model: Optional[BaseModel] = None,
    total_samples: int = 1,
    base_url: Optional[str] = None,
    verbose: bool = False,
) -> tuple[int, Optional[QueryResult]]:
    kwargs = sample_model_kwargs(
        model_names=model_names,
        temperatures=temperatures,
        max_tokens=max_tokens,
        reasoning_efforts=reasoning_efforts,
        model_sample_probs=model_sample_probs,
    )

                                                                          
    model_posteriors = None
    if model_sample_probs is not None and isinstance(model_names, list):
        model_posteriors = dict(zip(model_names, model_sample_probs))
        model_posteriors = {k: float(v) for k, v in model_posteriors.items()}
    if verbose:
        logger.info(f"==> SAMPLING: {idx + 1}/{total_samples} {list(kwargs.values())}")
    try_count = 0
    while try_count < MAX_RETRIES:
        try:
            result = query(
                msg=msg,
                system_msg=system_msg,
                msg_history=msg_history,
                output_model=output_model,
                model_posteriors=model_posteriors,
                base_url=base_url,
                **kwargs,
            )
            return idx, result
        except Exception as e:
            logger.error(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
            try_count += 1
            if try_count == MAX_RETRIES:
                                                      
                return idx, None
                                               
            time.sleep(1)
    return idx, None


def extract_between(
    content: str,
    start: str = "<json>",
    end: str = "</json>",
    return_dict: bool = True,
    fallback: bool = False,
) -> Optional[Union[str, dict]]:
    """Extract text from between start and end tags.

    Args:
        content (str): The input string containing CUDA code

    Returns:
        str: The extracted text, or None if no text is found
    """
    match = re.search(f"{start}\\s*(.*?)\\s*{end}", content, re.DOTALL)
    if match:
        matched_str = match.group(1).strip()
        if return_dict:
            return json.loads(matched_str)
        else:
            return matched_str

                                            
    if fallback:
        match = re.search("```\\s*(.*?)\\s*```", content, re.DOTALL)
        if match:
            matched_str = match.group(1).strip()
            if return_dict:
                return json.loads(matched_str)
            else:
                return matched_str
    return "none"
