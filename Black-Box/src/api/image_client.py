from __future__ import annotations

import base64
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import mimetypes

from openai import OpenAI

from src.config.settings import APISettings


class ImageEditClient:
    def __init__(self, config: APISettings):
        self.config = config
        self.provider = self._detect_provider(config.base_url, config.model)
        self.client = None
        
        # 只有真正走标准 OpenAI SDK 的才初始化官方 client
        if self.provider == "openai_compatible":
            self.client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout_seconds,
            )

    @staticmethod
    def _detect_provider(base_url: str, model: str) -> str:
        base = (base_url or "").lower()
        model_name = (model or "").lower()
        
        # 新增的路由分发逻辑
        if "stability.ai" in base:
            return "sd3"
        if "openrouter" in base:
            return "openrouter_flux"
        if "volces.com" in base or "doubao" in model_name:
            return "doubao"
            
        if "dashscope" in base or "aliyuncs.com" in base:
            if model_name.startswith("wanx"):
                return "wanx"
            return "dashscope"  # 默认对应 Qwen
            
        return "openai_compatible"

    def edit_image(self, input_image: Path, prompt: str) -> bytes:
        # 基于 provider 派发到对应的专属请求组装函数
        if self.provider == "dashscope":
            return self._edit_image_dashscope(input_image=input_image, prompt=prompt)
        elif self.provider == "wanx":
            return self._edit_image_wanx(input_image=input_image, prompt=prompt)
        elif self.provider == "openrouter_flux":
            return self._edit_image_openrouter(input_image=input_image, prompt=prompt)
        elif self.provider == "doubao":
            return self._edit_image_doubao(input_image=input_image, prompt=prompt)
        elif self.provider == "sd3":
            return self._edit_image_sd3(input_image=input_image, prompt=prompt)
            
        return self._edit_image_openai_compatible(input_image=input_image, prompt=prompt)

    # ---------------------------------------------------------
    # 1. 阿里万相 (Wanx) - Payload 与 Qwen 不同
    # ---------------------------------------------------------
    def _edit_image_wanx(self, input_image: Path, prompt: str) -> bytes:
        # 万相图生图端点
        endpoint_url = self._join_url(self.config.base_url, "api/v1/services/aigc/text2image/image-synthesis")
        
        payload = {
            "model": self.config.model,
            "input": {
                "prompt": prompt,
                "ref_img": self._image_to_data_url(input_image)
            },
            "parameters": dict(getattr(self.config, 'dashscope_parameters', {}))
        }
        
        # 注入重绘幅度参数 (万相的重绘参数名为 ref_strength)
        if hasattr(self.config, 'strength'):
            payload["parameters"]["ref_strength"] = getattr(self.config, 'strength', 0.15)

        # 强制要求同步返回 (X-DashScope-Async: disable)
        response_obj = self._http_post_json(
            url=endpoint_url,
            payload=payload,
            timeout_seconds=self.config.timeout_seconds,
            extra_headers={"X-DashScope-Async": "disable"} 
        )
        image_bytes = self._extract_image_bytes(response_obj)
        if not image_bytes:
            raise RuntimeError("Wanx response does not contain image bytes.")
        return image_bytes

    # ---------------------------------------------------------
    # 2. 字节豆包 (Doubao) - 需要精准控制 b64_json 与 strength
    # ---------------------------------------------------------
    def _edit_image_doubao(self, input_image: Path, prompt: str) -> bytes:
        endpoint_url = self._join_url(self.config.base_url, "api/v3/images/generations")
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "image": self._image_to_data_url(input_image),
            "response_format": "b64_json" # 字节强制要求
        }
        if hasattr(self.config, 'strength'):
            payload["strength"] = getattr(self.config, 'strength', 0.15)

        response_obj = self._http_post_json(
            url=endpoint_url,
            payload=payload,
            timeout_seconds=self.config.timeout_seconds,
        )
        image_bytes = self._extract_image_bytes(response_obj)
        if not image_bytes:
            raise RuntimeError("Doubao response does not contain image bytes.")
        return image_bytes

    # ---------------------------------------------------------
    # 3. OpenRouter FLUX - 必须伪装成 Chat Completion 多模态
    # ---------------------------------------------------------
    def _edit_image_openrouter(self, input_image: Path, prompt: str) -> bytes:
        endpoint_url = self._join_url(self.config.base_url, "api/v1/chat/completions")
        
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": self._image_to_data_url(input_image)}}
                    ]
                }
            ],
            "modalities": ["image"] # OpenRouter 专属出图指令
        }
        
        response_obj = self._http_post_json(
            url=endpoint_url,
            payload=payload,
            timeout_seconds=self.config.timeout_seconds,
        )
        image_bytes = self._extract_image_bytes(response_obj)
        if not image_bytes:
            raise RuntimeError("OpenRouter FLUX response does not contain image bytes.")
        return image_bytes

    # ---------------------------------------------------------
    # 4. Stability AI (SD3) - 必须使用 multipart/form-data
    # ---------------------------------------------------------
    def _edit_image_sd3(self, input_image: Path, prompt: str) -> bytes:
        endpoint_url = self._join_url(self.config.base_url, "v2beta/stable-image/generate/sd3")
        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
        body = []

        def add_field(name: str, value: str):
            body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode('utf-8'))

        def add_file(name: str, filename: str, content: bytes):
            mime = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n".encode('utf-8'))
            body.append(content)
            body.append(b"\r\n")

        add_field("prompt", prompt)
        add_field("mode", "image-to-image")
        add_field("model", self.config.model or "sd3.5-large")
        if hasattr(self.config, 'strength'):
            add_field("strength", str(getattr(self.config, 'strength', 0.15)))

        add_file("image", input_image.name, input_image.read_bytes())
        body.append(f"--{boundary}--\r\n".encode('utf-8'))
        req_data = b"".join(body)

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        }
        req = urllib.request.Request(url=endpoint_url, data=req_data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                text = response.read().decode(charset, errors="replace")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SD3 HTTP {exc.code}: {err_body}") from exc

        data = json.loads(text)
        image_bytes = self._extract_image_bytes(data)
        if not image_bytes:
            raise RuntimeError("SD3 response does not contain image bytes.")
        return image_bytes

    # ---------------------------------------------------------
    # 5. 保留原有的 OpenAI Compatible 和 Qwen 逻辑
    # ---------------------------------------------------------
    def _edit_image_openai_compatible(self, input_image: Path, prompt: str) -> bytes:
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized.")
        request: dict[str, str] = {
            "model": self.config.model,
            "prompt": prompt,
        }
        if self.config.size:
            request["size"] = self.config.size
        if self.config.quality:
            request["quality"] = self.config.quality

        with input_image.open("rb") as image_file:
            response = self.client.images.edit(image=image_file, **request)

        b64_json = response.data[0].b64_json
        if not b64_json:
            raise RuntimeError("`images.edit` returned empty image data.")
        return base64.b64decode(b64_json)

    def _edit_image_dashscope(self, input_image: Path, prompt: str) -> bytes:
        endpoint_url = self._join_url(self.config.base_url, getattr(self.config, 'dashscope_endpoint', 'api/v1/services/aigc/text2image/image-synthesis'))
        parameters = dict(getattr(self.config, 'dashscope_parameters', {}))
        if self.config.size and "size" not in parameters:
            parameters["size"] = self.config.size
        if "n" not in parameters:
            parameters["n"] = 1

        payload = {
            "model": self.config.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": self._image_to_data_url(input_image)},
                            {"text": prompt},
                        ],
                    }
                ]
            },
            "parameters": parameters,
            "seed": 42,
        }
        response_obj = self._http_post_json(
            url=endpoint_url,
            payload=payload,
            timeout_seconds=self.config.timeout_seconds,
        )
        image_bytes = self._extract_image_bytes(response_obj)
        if image_bytes is None:
            raise RuntimeError("DashScope response does not contain image bytes.")
        return image_bytes

    # ---------------------------------------------------------
    # 工具函数区：增加了 extra_headers 支持
    # ---------------------------------------------------------
    def _http_post_json(
        self,
        *,
        url: str,
        payload: dict[str, object],
        timeout_seconds: float,
        extra_headers: dict[str, str] | None = None
    ) -> dict[str, object]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        if extra_headers:
            headers.update(extra_headers)
            
        req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                text = response.read().decode(charset, errors="replace")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {err_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error: {exc}") from exc

        data = json.loads(text)
        if not isinstance(data, dict):
            raise RuntimeError("Response root must be a JSON object.")
        return data

    @staticmethod
    def _join_url(base_url: str, endpoint: str) -> str:
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        return urllib.parse.urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        suffix = image_path.suffix.lower().lstrip(".") or "png"
        raw = image_path.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:image/{suffix};base64,{encoded}"

    def _extract_image_bytes(self, payload: object) -> bytes | None:
        if isinstance(payload, list):
            for item in payload:
                image_bytes = self._extract_image_bytes(item)
                if image_bytes is not None:
                    return image_bytes
            return None

        if not isinstance(payload, dict):
            return None

        for key in ("b64_json", "image_base64", "base64", "result"):
            value = payload.get(key)
            if isinstance(value, str):
                maybe = self._decode_base64(value)
                if maybe is not None:
                    return maybe

        for key in ("url", "image_url", "image"):
            value = payload.get(key)
            if isinstance(value, str):
                if value.startswith(("http://", "https://")):
                    with urllib.request.urlopen(value, timeout=self.config.timeout_seconds) as response:
                        return response.read()
                if value.startswith("data:image/") and ";base64," in value:
                    encoded = value.split(";base64,", maxsplit=1)[1]
                    maybe = self._decode_base64(encoded)
                    if maybe is not None:
                        return maybe
                maybe = self._decode_base64(value)
                if maybe is not None:
                    return maybe

        for value in payload.values():
            image_bytes = self._extract_image_bytes(value)
            if image_bytes is not None:
                return image_bytes
        return None

    @staticmethod
    def _decode_base64(value: str) -> bytes | None:
        cleaned = value.strip()
        if len(cleaned) < 64:
            return None
        if len(cleaned) % 4 != 0:
            return None
        try:
            return base64.b64decode(cleaned)
        except Exception:
            return None