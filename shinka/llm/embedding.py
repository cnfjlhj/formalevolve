import os
import json
import openai
import google.generativeai as genai
import pandas as pd
from typing import Union, List, Optional, Tuple
import numpy as np
import logging
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


M = 1_000_000

OPENAI_EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
]

AZURE_EMBEDDING_MODELS = [
    "azure-text-embedding-3-small",
    "azure-text-embedding-3-large",
]

GEMINI_EMBEDDING_MODELS = [
    "gemini-embedding-exp-03-07",
    "gemini-embedding-001",
]

HF_EMBEDDING_PREFIXES = ("hf:", "huggingface:")

OPENAI_EMBEDDING_COSTS = {
    "text-embedding-3-small": 0.02 / M,
    "text-embedding-3-large": 0.13 / M,
}

                                                              
GEMINI_EMBEDDING_COSTS = {
    "gemini-embedding-exp-03-07": 0.0 / M,                                  
    "gemini-embedding-001": 0.0 / M,                         
}

def _detect_e5_http_server(embed_base_url: str) -> Optional[str]:
    """
    Detect the lightweight "E5 Embedding Server" HTTP API (non-OpenAI-compatible).

    Expected shape (FastAPI):
      - GET  /openapi.json  -> info.title == "E5 Embedding Server"
      - POST /embed         -> {"texts":[...], "normalize":true} -> {"embeddings":[...], "dim":...}

    Returns:
      The normalized base URL (no trailing slash) if detected, else None.
    """
    base = str(embed_base_url or "").strip()
    if not base:
        return None

    base = base.rstrip("/")
    candidates = [base]
    if base.endswith("/v1"):
        candidates.append(base[: -len("/v1")])

    timeout_s = float(os.environ.get("E5_HTTP_EMBED_DETECT_TIMEOUT", "0.5"))
    for cand in dict.fromkeys(candidates):                           
        url = f"{cand.rstrip('/')}/openapi.json"
        try:
            with urlopen(url, timeout=timeout_s) as resp:
                if int(getattr(resp, "status", 200) or 200) != 200:
                    continue
                spec = json.loads(resp.read().decode("utf-8"))
            title = str(((spec.get("info") or {}).get("title") or "")).strip()
            paths = spec.get("paths") or {}
            if title == "E5 Embedding Server" and "/embed" in paths:
                return cand.rstrip("/")
        except Exception:
            continue

    return None

def get_client_model(model_name: str) -> tuple[Union[openai.OpenAI, str], str]:
    if any(model_name.startswith(p) for p in HF_EMBEDDING_PREFIXES):
                                                                                                             
        model_to_use = model_name.split(":", 1)[1].strip()
        if not model_to_use:
            raise ValueError("Invalid HuggingFace embedding model spec: expected 'hf:<model_or_path>'")
        return "huggingface", model_to_use

    if model_name in OPENAI_EMBEDDING_MODELS:
                                                                        
                                                                         
        embed_base_url = os.environ.get("OPENAI_EMBED_BASE_URL")
        if embed_base_url:
            client = openai.OpenAI(base_url=embed_base_url)
        else:
            client = openai.OpenAI()
        model_to_use = model_name
    elif model_name in AZURE_EMBEDDING_MODELS:
                                      
        model_to_use = model_name.split("azure-")[-1]
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
        )
    elif model_name in GEMINI_EMBEDDING_MODELS:
                              
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set for Gemini models")
        genai.configure(api_key=api_key)
        client = "gemini"                                    
        model_to_use = model_name
    else:
                                                                                                  
                                                                                    
        embed_base_url = os.environ.get("OPENAI_EMBED_BASE_URL")
        if embed_base_url:
                                                                                         
            e5_base_url = _detect_e5_http_server(embed_base_url)
            if e5_base_url:
                client = "e5_http"
                model_to_use = e5_base_url
                logger.info(
                    "Using E5 HTTP embedding server (POST /embed) via "
                    f"OPENAI_EMBED_BASE_URL={embed_base_url}"
                )
            else:
                client = openai.OpenAI(
                    base_url=embed_base_url,
                    api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                )
                model_to_use = model_name
                logger.info(
                    f"Using custom OpenAI-compatible embedding model '{model_name}' via OPENAI_EMBED_BASE_URL={embed_base_url}"
                )
        else:
                                                                                                  
            client = openai.OpenAI()
            model_to_use = model_name
            logger.warning(
                f"Unknown embedding model '{model_name}' without OPENAI_EMBED_BASE_URL; "
                "falling back to default OpenAI client (may fail)."
            )

    return client, model_to_use


class EmbeddingClient:
    def __init__(
        self, model_name: str = "text-embedding-3-small", verbose: bool = False
    ):
        """
        Initialize the EmbeddingClient.

        Args:
            model (str): The OpenAI, Azure, or Gemini embedding model name to use.
        """
        self.client, self.model = get_client_model(model_name)
        self.model_name = model_name
        self.verbose = verbose
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_device = None

        if self.client == "huggingface":
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "HuggingFace embedding backend requires `transformers` and `torch`. "
                    "Install with: pip install transformers torch"
                ) from e

            device_str = os.environ.get("HF_EMBEDDING_DEVICE")
            if not device_str:
                device_str = "cuda" if torch.cuda.is_available() else "cpu"

            self._hf_device = torch.device(device_str)
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._hf_model = AutoModel.from_pretrained(self.model)
            self._hf_model.eval()
            self._hf_model.to(self._hf_device)

    def get_embedding(
        self, code: Union[str, List[str]]
    ) -> Union[Tuple[List[float], float], Tuple[List[List[float]], float]]:
        """
        Computes the text embedding for a CUDA kernel string.

        Args:
            code (str, list[str]): The CUDA kernel code as a string or list
                of strings.

        Returns:
            list: Embedding vector for the kernel code or None if an error
                occurs.
        """
        if isinstance(code, str):
            code = [code]
            single_code = True
        else:
            single_code = False

                                                  
        if self.client == "huggingface":
            import torch
            import torch.nn.functional as F

            assert self._hf_tokenizer is not None and self._hf_model is not None and self._hf_device is not None
            max_length = int(os.environ.get("HF_EMBEDDING_MAX_LENGTH", "512"))

            try:
                inputs = self._hf_tokenizer(
                    code,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._hf_device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._hf_model(**inputs)
                    last_hidden = outputs.last_hidden_state             
                    attn = inputs.get("attention_mask")
                    if attn is None:
                        pooled = last_hidden.mean(dim=1)
                    else:
                        mask = attn.unsqueeze(-1).float()
                        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    pooled = F.normalize(pooled, p=2, dim=1)

                embeddings = pooled.detach().cpu().tolist()
                cost = 0.0
                if single_code:
                    return embeddings[0] if embeddings else [], cost
                return embeddings, cost
            except Exception as e:
                logger.error(f"Error getting HuggingFace embedding: {e}")
                if single_code:
                    return [], 0.0
                return [[]], 0.0

                                                                                  
        if self.client == "e5_http":
            base_url = str(self.model or "").rstrip("/")
            if not base_url:
                if single_code:
                    return [], 0.0
                return [[]], 0.0

            payload = {"texts": code, "normalize": True}
            data = json.dumps(payload).encode("utf-8")
            timeout_s = float(os.environ.get("E5_HTTP_EMBED_TIMEOUT", "30"))
            try:
                req = Request(
                    f"{base_url}/embed",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req, timeout=timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                obj = json.loads(body)
                embeddings = obj.get("embeddings") or []
                if single_code:
                    return embeddings[0] if embeddings else [], 0.0
                return embeddings, 0.0
            except Exception as e:
                logger.error(f"Error getting E5 HTTP embedding: {e}")
                if single_code:
                    return [], 0.0
                return [[]], 0.0

                              
        if self.model_name in GEMINI_EMBEDDING_MODELS:
            try:
                embeddings = []
                total_tokens = 0
                
                for text in code:
                    result = genai.embed_content(
                        model=f"models/{self.model}",
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(result['embedding'])
                    total_tokens += len(text.split())
                
                cost = total_tokens * GEMINI_EMBEDDING_COSTS.get(self.model, 0.0)
                
                if single_code:
                    return embeddings[0] if embeddings else [], cost
                else:
                    return embeddings, cost
            except Exception as e:
                logger.error(f"Error getting Gemini embedding: {e}")
                if single_code:
                    return [], 0.0
                else:
                    return [[]], 0.0
                                                         
        try:
            response = self.client.embeddings.create(
                model=self.model, input=code, encoding_format="float"
            )
            usage = getattr(response, "usage", None)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            cost_per_token = float(OPENAI_EMBEDDING_COSTS.get(self.model, 0.0))
            cost = total_tokens * cost_per_token
                                             
            if single_code:
                return response.data[0].embedding, cost
            else:
                return [d.embedding for d in response.data], cost
        except Exception as e:
            logger.info(f"Error getting embedding: {e}")
            if single_code:
                return [], 0.0
            else:
                return [[]], 0.0

    def get_column_embedding(
        self,
        df: pd.DataFrame,
        column_name: Union[str, List[str]],
    ) -> pd.DataFrame:
        """
        Computes the text embedding for a batch of CUDA kernel strings.

        Args:
            df (pd.DataFrame): A pandas DataFrame with the column to embed.
            column_name (str, list): The name of the columns to embed.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the column to embed.
        """
        if isinstance(column_name, str):
            column_name = [column_name]

        for column_name in column_name:
            model_name_str = self.model.replace("-", "_")
            new_col_name = f"{column_name}_embedding_{model_name_str}"
            df[new_col_name] = df[column_name].apply(
                lambda x: self.get_embedding(x),
            )
        return df

    def get_closest_k_neighbors(
        self,
        new_str_query: str,
        embeddings: list,
        top_k: Union[int, str] = 5,
    ) -> tuple[list, list]:
        """Get k closest neighbors from the embeddings list

        Args:
            new_str_query: The string to get the closest neighbors for.
            embeddings: The list of embeddings to compare against.
            top_k: The number of closest neighbors to return.

        Returns:
            A tuple of the top k indices and the top k similarities.
        """
                                         
        new_embedding, _ = self.get_embedding(new_str_query)

        if not new_embedding:                                     
            return [], []

                                  
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

                                                             
        similarities = [
            cosine_similarity(new_embedding, embedding) for embedding in embeddings
        ]

                                                
        if top_k == "random":
            if len(similarities) < 5:
                top_idx = np.random.choice(
                    len(similarities), size=len(similarities), replace=False
                )
            else:
                top_idx = np.random.choice(len(similarities), size=5, replace=False)
            similarities_subset = [similarities[i] for i in top_idx]
            return top_idx.tolist(), similarities_subset
        elif isinstance(top_k, int):
            top_idx = np.argsort(similarities)[-top_k:]
            similarities_subset = [similarities[i] for i in top_idx]
            return top_idx[::-1].tolist(), similarities_subset[::-1]
        else:
            raise ValueError("top_k must be an int or 'random'")

    def get_dim_reduction(
        self,
        embeddings: list,
        method: str = "pca",
        dims: int = 2,
    ):
        """Performs dimensionality reduction on a list of embeddings using
        various methods.

        Args:
            embeddings: List of embedding vectors
            method: Dimensionality reduction method ('pca', 'umap', or 'tsne')
            dims: Number of dimensions to reduce to

        Returns:
            The transformed embeddings in reduced dimensionality
        """
        if isinstance(embeddings, pd.Series):
            embeddings = embeddings.tolist()

                                               
        X = np.array(embeddings) if isinstance(embeddings, list) else embeddings
                                                         
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if method.lower() == "pca":
            from sklearn.decomposition import PCA

            model = PCA(n_components=dims)
            return model.fit_transform(X)
        elif method.lower() == "umap":
            from umap import UMAP

            model = UMAP(n_components=dims, random_state=42)
            return model.fit_transform(X)
        elif method.lower() == "tsne":
            from sklearn.manifold import TSNE

            model = TSNE(n_components=dims, random_state=42)
            return model.fit_transform(X)
        else:
            raise ValueError("Method must be one of: 'pca', 'umap', 'tsne'")

    def get_embedding_clusters(
        self,
        embeddings: list,
        num_clusters: int = 4,
        verbose: bool = False,
    ) -> list:
        """
        Performs clustering on a list of embeddings using Gaussian Mixture Model.

        Args:
            embeddings: List of embedding vectors
            num_clusters: Number of clusters to form with GMM.
            top_k_candidates: Number of top kernels to select per cluster.
            verbose: If True, prints detailed cluster information.

        Returns:
            pd.DataFrame: A DataFrame with top candidate kernels from each
            cluster.
        """
        from sklearn.mixture import GaussianMixture

                                                              
        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        gmm.fit(embeddings)
        clusters = gmm.predict(embeddings)

                                                         
        if verbose:
            logger.info(
                f"GMM {num_clusters} Clusters ==> Got {len(embeddings)} "
                f"embeddings with cluster assignments:"
            )
            num_members = pd.Series(clusters).value_counts()
            logger.info(num_members)

        return clusters

    def plot_reduced_embeddings(
        self,
        embeddings: list,
        method: str = "pca",
        num_dims: int = 3,
        title="Embedding",
        cluster_ids: Optional[list] = None,
        cluster_label: str = "Cluster",
        patch_type: Optional[list] = None,
    ):
        transformed = self.get_dim_reduction(embeddings, method, num_dims)

        if num_dims == 2:
            fig, ax = plot_2d_scatter(
                transformed, title, cluster_ids, cluster_label, patch_type
            )
        elif num_dims == 3:
            fig, ax = plot_3d_scatter(
                transformed, title, cluster_ids, cluster_label, patch_type
            )
        else:
            raise ValueError(f"Invalid number of dimensions: {num_dims}")

        return fig, ax


def plot_2d_scatter(
    transformed: np.ndarray,
    title: str = "Embedding",
    cluster_ids: Optional[list] = None,
    cluster_label: str = "Cluster",
    patch_type: Optional[list] = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

                                                              
    fig, ax = plt.subplots(figsize=(10, 7))

                                      
    if cluster_ids is not None:
        original_unique_ids, cluster_ids_for_coloring = np.unique(
            cluster_ids, return_inverse=True
        )
        num_distinct_colors = len(original_unique_ids)
                                                                             
                                                                       
                                           
    else:
        cluster_ids_for_coloring = np.zeros(transformed.shape[0])
        original_unique_ids = [
            0
        ]                                                                  
        num_distinct_colors = 1

                              
    base_colors = [
        "green",
        "red",
        "blue",
        "yellow",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "cyan",
    ]
    if num_distinct_colors > 0:
        multiplier = (num_distinct_colors - 1) // len(base_colors) + 1
        extended_colors = base_colors * multiplier
        colors_for_cmap = extended_colors[:num_distinct_colors]
    else:                                                            
        colors_for_cmap = ["blue"]

    cmap = ListedColormap(colors_for_cmap)

    marker_shapes = ["o", "s", "^", "P", "X", "D", "v", "<", ">"]

    if patch_type is not None:
        patch_type_array = np.array(patch_type)
        unique_patches = np.unique(patch_type_array)

        for i, patch_val in enumerate(unique_patches):
            patch_mask = patch_type_array == patch_val
            current_marker = marker_shapes[i % len(marker_shapes)]

            c_val_scatter = None
            cmap_val_scatter = (
                None                                                      
            )
            if cluster_ids is not None:
                c_val_scatter = cluster_ids_for_coloring[patch_mask]
                cmap_val_scatter = cmap

            label_text = str(patch_val)

            scatter_args = {
                "marker": current_marker,
                "alpha": 0.6,
                "s": 100,
                "label": label_text,
            }
            if c_val_scatter is not None:                       
                scatter_args["c"] = c_val_scatter
                scatter_args["cmap"] = cmap_val_scatter

            ax.scatter(
                transformed[patch_mask, 0],       
                transformed[patch_mask, 1],       
                **scatter_args,
            )
    else:                 
        c_val_scatter_else = None
        if cluster_ids is not None:
            c_val_scatter_else = (
                cluster_ids_for_coloring                                  
            )

                                                                   

        scatter_args_else = {"marker": "o", "alpha": 0.6, "s": 100}
        if (
            c_val_scatter_else is not None
        ):                                                   
            scatter_args_else["c"] = c_val_scatter_else
            scatter_args_else["cmap"] = cmap                                 

        ax.scatter(
            transformed[:, 0],       
            transformed[:, 1],       
            **scatter_args_else,
        )

                                                
    ax.set_xlabel("1st Latent Dim.", fontsize=20)
    ax.set_ylabel("2nd Latent Dim.", fontsize=20)
    ax.set_title(title, fontsize=30)

                                 
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

                                       
    if (
        cluster_ids is not None
    ):                                                                  
        try:
                                                                             
                                                      
                                                                
            ax.scatter(
                transformed[:, 0],
                transformed[:, 1],
                c=cluster_ids_for_coloring,                                 
                cmap=cmap,                     
                s=0,
                alpha=0,
            )
                                                                            
                                                  
                                                                               
                                                             
                                          
                                                
                                                
                                 
                   
                                                                
        except Exception:
            pass                 

    if patch_type is not None:
                                                        
        legend_handles = []
        unique_patches_for_legend = np.unique(np.array(patch_type))
        for i, patch_val in enumerate(unique_patches_for_legend):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker_shapes[i % len(marker_shapes)],
                    color="black",
                    label=str(patch_val),
                    linestyle="None",
                    markersize=10,
                )
            )
        if legend_handles:
            ax.legend(handles=legend_handles, title="Patch Types", loc="best")

    fig.tight_layout()
                                                         
                                                   
                                         

    return fig, ax


def plot_3d_scatter(
    transformed: np.ndarray,
    title: str = "Embedding",
    cluster_ids: Optional[list] = None,
    cluster_label: str = "Cluster",
    patch_type: Optional[list] = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap

                                                              
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

                                      
    if cluster_ids is not None:
        original_unique_ids, cluster_ids_for_coloring = np.unique(
            cluster_ids, return_inverse=True
        )
        num_distinct_colors = len(original_unique_ids)
    else:
        cluster_ids_for_coloring = np.zeros(transformed.shape[0])
        original_unique_ids = [0]
        num_distinct_colors = 1

                              
    base_colors = [
        "green",
        "red",
        "blue",
        "yellow",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "cyan",
    ]
    if num_distinct_colors > 0:
        multiplier = (num_distinct_colors - 1) // len(base_colors) + 1
        extended_colors = base_colors * multiplier
        colors_for_cmap = extended_colors[:num_distinct_colors]
    else:
        colors_for_cmap = ["blue"]

    cmap = ListedColormap(colors_for_cmap)

    marker_shapes = ["o", "s", "^", "P", "X", "D", "v", "<", ">"]

    if patch_type is not None:
        patch_type_array = np.array(patch_type)
        unique_patches = np.unique(patch_type_array)

        for i, patch_val in enumerate(unique_patches):
            patch_mask = patch_type_array == patch_val
            current_marker = marker_shapes[i % len(marker_shapes)]

            c_val_scatter = None
            cmap_val_scatter = None
            if cluster_ids is not None:
                c_val_scatter = cluster_ids_for_coloring[patch_mask]
                cmap_val_scatter = cmap

            label_text = str(patch_val)

            scatter_args = {
                "marker": current_marker,
                "alpha": 0.6,
                "s": 20,                                  
                "label": label_text,
                                       
            }
            if c_val_scatter is not None:
                scatter_args["c"] = c_val_scatter
                scatter_args["cmap"] = cmap_val_scatter

            scatter = ax.scatter(
                transformed[patch_mask, 0],       
                transformed[patch_mask, 1],       
                transformed[patch_mask, 2],       
                **scatter_args,
            )
    else:                 
        c_val_scatter_else = None
        if cluster_ids is not None:
            c_val_scatter_else = cluster_ids_for_coloring

        scatter_args_else = {
            "marker": "o",
            "alpha": 0.6,
            "s": 20,                                  
                                   
        }
        if c_val_scatter_else is not None:
            scatter_args_else["c"] = c_val_scatter_else
            scatter_args_else["cmap"] = cmap

        scatter = ax.scatter(
            transformed[:, 0],       
            transformed[:, 1],       
            transformed[:, 2],       
            **scatter_args_else,
        )

                                                
    ax.set_xlabel("1st Latent Dim.", labelpad=-15, fontsize=8)
    ax.set_ylabel("2nd Latent Dim.", labelpad=-15, fontsize=8)
    ax.set_zlabel(
        "3rd Latent Dim.", labelpad=-17, rotation=90, fontsize=8
    )                                        
    ax.set_title(title, y=0.95)

                                       
    if cluster_ids is not None:                        
        try:
            temp_scatter_for_colorbar = ax.scatter(
                transformed[:, 0],
                transformed[:, 1],
                transformed[:, 2],
                c=cluster_ids_for_coloring,                     
                cmap=cmap,
                s=0,
                alpha=0,
            )
                                                  
                                                                               
                
                                          
                                                                                      
                   
                                                   
        except Exception:
            pass                 

    if patch_type is not None:
                                                        
        legend_handles_3d = []
        unique_patches_for_legend_3d = np.unique(np.array(patch_type))
        for i, patch_val in enumerate(unique_patches_for_legend_3d):
            legend_handles_3d.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker_shapes[i % len(marker_shapes)],
                    color="black",
                    label=str(patch_val),
                    linestyle="None",
                    markersize=10,
                )
            )
        if legend_handles_3d:
            ax.legend(
                handles=legend_handles_3d,
                title="Patch Types",
                loc="best",
                bbox_to_anchor=(0.9, 0.5),
            )

                                                    
    ax.view_init(elev=20, azim=45)

                                                                           
                     
    plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.05)
    fig.tight_layout()
    return fig, ax
