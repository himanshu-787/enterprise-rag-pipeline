#!/usr/bin/env python3
"""Downloads or generates the FastAPI technical documentation corpus."""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.config import settings

DOCS_CORPUS = {
    "index.md": """# FastAPI Framework Overview

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.

## Key Features

- **Fast**: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic). One of the fastest Python frameworks available.
- **Fast to code**: Increase the speed to develop features by about 200% to 300%.
- **Fewer bugs**: Reduce about 40% of human (developer) induced errors.
- **Intuitive**: Great editor support. Completion everywhere. Less time debugging.
- **Easy**: Designed to be easy to use and learn. Less time reading docs.
- **Short**: Minimize code duplication. Multiple features from each parameter declaration.
- **Robust**: Get production-ready code. With automatic interactive documentation.
- **Standards-based**: Based on (and fully compatible with) the open standards for APIs: OpenAPI (previously known as Swagger) and JSON Schema.

## Technical Foundations

FastAPI stands on the shoulders of giants:
- **Starlette** for the web routing, ASGI concurrency, and event handling.
- **Pydantic** for data validation, serialization, and schema definition.
- **Uvicorn** as the lightning-fast ASGI server implementation using `uvloop` and `httptools`.
""",

    "tutorial/first_steps.md": """# First Steps with FastAPI

Creating your first FastAPI application requires just a few lines of Python code.

## The Simplest FastAPI App

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
```

### Explanation of Components

1. **Import `FastAPI`**: `FastAPI` is a Python class that provides all the functionality for your API.
2. **Create an instance**: `app = FastAPI()`. This is the main point of interaction to create all your API endpoints.
3. **Define a path operation decorator**: `@app.get("/")` tells FastAPI that the function right below is in charge of handling requests that go to the path `/` using a `get` HTTP operation.
4. **Define the path operation function**: `async def root():` is the async function called whenever a matching HTTP GET request arrives.
5. **Return content**: You can return a `dict`, `list`, singular values as `str`, `int`, or Pydantic models. They will be automatically converted to JSON.

/// tip
You can also define standard synchronous functions with `def root():` if you are not using asynchronous I/O libraries.
///
""",

    "tutorial/path_params.md": """# Path Parameters

You can declare path "parameters" or "variables" with the same syntax used by Python format strings.

## Basic Path Parameter Example

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

The value of the path parameter `item_id` will be passed to your function as the argument `item_id`.

## Path Parameters with Types

In the example above, `item_id` is declared to be an `int`. This gives you:
- **Editor support**: Autocomplete and type checks.
- **Data parsing**: FastAPI automatically parses the string from HTTP into a Python `int`.
- **Data validation**: If someone requests `/items/foo`, FastAPI returns an HTTP 422 Unprocessable Entity error explaining that `item_id` must be an integer.

## Path Order Matters

When creating path operations, you may have fixed paths before dynamic paths:

```python
@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}


@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}
```

Because path operations are evaluated in order, declare `/users/me` before `/users/{user_id}` so that `/users/me` isn't matched as a user ID `"me"`.
""",

    "tutorial/query_params.md": """# Query Parameters

When you declare function parameters that are not part of the path parameters, they are automatically interpreted as "query" parameters.

## Declaration and Defaults

The query is the set of key-value pairs that go after the `?` in a URL, separated by `&` characters.

```python
from fastapi import FastAPI

app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```

Here, `skip` and `limit` are query parameters with default values of `0` and `10` respectively.
A URL like `http://127.0.0.1:8000/items/?skip=0&limit=2` will pass `skip=0` and `limit=2`.

## Optional Query Parameters

You can declare optional query parameters by setting their default to `None`:

```python
from typing import Optional
from fastapi import FastAPI

app = FastAPI()


@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Optional[str] = None, short: bool = False):
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item
```

## Boolean Type Conversion

FastAPI automatically parses boolean query parameters. Any of `1`, `True`, `true`, `on`, `yes` will convert to `True`.
""",

    "tutorial/request_body.md": """# Request Body and Pydantic Models

When you need to send data from a client to your API, you send it as a request body.

## Declaring Request Bodies with Pydantic

To declare a request body, you use Pydantic models with all their power and benefits.

```python
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


app = FastAPI()


@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

## Results of Pydantic Integration

With just that Python type declaration, FastAPI will:
1. Read the body of the request as JSON.
2. Convert the corresponding types (if needed).
3. Validate the data. If the data is invalid, it will return a clean and clear HTTP 422 error.
4. Give you the received data in the parameter `item` with full IDE type hinting.
5. Generate JSON Schema definitions for your models in the OpenAPI specification.
""",

    "tutorial/query_params_validation.md": """# Query Parameters and String Validations

FastAPI allows you to declare additional information and validation for your parameters using `Query`.

## Validation Constraints

You can enforce length constraints, regular expressions, and metadata:

```python
from typing import Optional
from fastapi import FastAPI, Query

app = FastAPI()


@app.get("/items/")
async def read_items(
    q: Optional[str] = Query(
        default=None,
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9_-]+$",
        title="Search Query",
        description="Query string for the items to search in the database",
        deprecated=False,
    )
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results
```

## Required Query Parameters with Ellipsis

When you need to make a parameter mandatory while still using `Query` for validation, use `...` (Ellipsis) as the default:

```python
@app.get("/items/required")
async def read_items_required(q: str = Query(..., min_length=3)):
    return {"q": q}
```

## Query Parameter List / Multiple Values

To accept multiple query parameter values (e.g. `?q=foo&q=bar`), declare the type as `list[str]`:

```python
@app.get("/items/multi")
async def read_items_multi(q: list[str] = Query(default=["foo", "bar"])):
    return {"q": q}
```
""",

    "tutorial/dependencies.md": """# Dependency Injection System

FastAPI has a very powerful and intuitive Dependency Injection system. It allows you to share logic, enforce database connections, implement security, and authenticate users.

## What is Dependency Injection?

In software engineering, "dependency injection" is a technique whereby an object receives other objects that it depends on. In FastAPI, it enables:
- Shared database connections or sessions
- Authentication and authorization checks
- Common query parameters across multiple endpoints
- Role-based access control (RBAC)

## Creating and Using a First Dependency

```python
from typing import Optional
from fastapi import Depends, FastAPI

app = FastAPI()


async def common_parameters(q: Optional[str] = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}


@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons


@app.get("/users/")
async def read_users(commons: dict = Depends(common_parameters)):
    return commons
```

### How `Depends` Works

When a request arrives, FastAPI:
1. Calls the dependency function `common_parameters` with the appropriate parameters extracted from the request.
2. Takes the return value of `common_parameters`.
3. Injects that return value into the path operation function as the argument `commons`.
""",

    "tutorial/sub_dependencies.md": """# Sub-dependencies and Hierarchical Trees

You can declare dependencies that themselves declare dependencies. FastAPI handles solving the entire dependency graph automatically.

## Hierarchical Dependency Example

```python
from typing import Optional
from fastapi import Cookie, Depends, FastAPI

app = FastAPI()


def query_extractor(q: Optional[str] = None):
    return q


def query_or_cookie_extractor(
    q: str = Depends(query_extractor),
    last_query: Optional[str] = Cookie(default=None),
):
    if not q:
        return last_query
    return q


@app.get("/items/")
async def read_query(query_or_default: str = Depends(query_or_cookie_extractor)):
    return {"q_or_cookie": query_or_default}
```

## Dependency Caching with `use_cache`

By default, if a dependency is shared multiple times across a single request (e.g. `get_current_user` used in both route and sub-dependency), FastAPI caches the result for that request and only calls the dependency once:

```python
async def get_db():
    db = DBSession()
    try:
        yield db
    finally:
        db.close()


# Cached per request by default:
@app.get("/users/")
async def get_users(db: DBSession = Depends(get_db, use_cache=True)):
    pass
```
""",

    "tutorial/dependencies_with_yield.md": """# Dependencies with `yield` and Resource Teardown

FastAPI supports dependencies that do some extra steps after finishing. To do this, use `yield` instead of `return`.

## Database Session Lifecycle Pattern

```python
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

app = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/items/{item_id}")
def read_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ItemModel).filter(ItemModel.id == item_id).first()
    return item
```

### Execution Flow with `yield`

1. Code prior to `yield` is executed **before** the path operation function runs.
2. The yielded value is injected into the endpoint function.
3. The endpoint function processes and returns the response.
4. The code after `yield` executes **after** the response is sent or if an exception was raised, ensuring clean resource teardown (e.g., closing database connections, cleaning temporary files).
""",

    "tutorial/security_oauth2.md": """# Security and Authentication: OAuth2 with Password and Bearer

FastAPI provides high-level security utilities based on OpenAPI standards.

## Implementing OAuth2PasswordBearer

```python
from typing import Optional
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class User(BaseModel):
    username: str
    email: Optional[str] = None


async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_decode_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
```

## Token Endpoint

You implement the login endpoint using `OAuth2PasswordRequestForm`:

```python
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict or form_data.password != user_dict["password"]:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": user_dict["username"], "token_type": "bearer"}
```
""",

    "tutorial/cors_middleware.md": """# CORS (Cross-Origin Resource Sharing) and Middleware

FastAPI makes it trivial to configure Cross-Origin Resource Sharing (CORS) using `CORSMiddleware`.

## Configuring CORSMiddleware

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "https://mycompany.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def main():
    return {"message": "CORS enabled"}
```

## Custom Middleware for Timing and Request ID

You can add custom middleware using the `@app.middleware("http")` decorator:

```python
import time
from fastapi import Request


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time * 1000:.2f}ms"
    return response
```
""",

    "tutorial/background_tasks.md": """# Background Tasks

FastAPI provides an easy way to define tasks that will run after returning a response. This is useful for operations that need to happen after a request, but where the client does not need to wait for the operation before receiving the response.

## BackgroundTasks Declaration

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()


def write_notification(email: str, message: str = ""):
    with open("log.txt", mode="a") as email_file:
        content = f"notification for {email}: {message}\n"
        email_file.write(content)


@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        write_notification, email, message="some notification message"
    )
    return {"message": "Notification sent in the background"}
```

### Dependency Injection with BackgroundTasks

You can also inject `BackgroundTasks` directly into dependencies:

```python
def log_operation(background_tasks: BackgroundTasks):
    background_tasks.add_task(audit_log, "operation completed")
```
""",

    "tutorial/handling_errors.md": """# Handling Errors and Custom Exceptions

FastAPI uses `HTTPException` to return HTTP error responses with status codes and structured details.

## Raising `HTTPException`

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

items = {"foo": "The Foo Fighters"}


@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "Custom header value"},
        )
    return {"item": items[item_id]}
```

## Custom Exception Handlers

You can register custom exception handlers for domain-specific exceptions:

```python
from fastapi import Request
from fastapi.responses import JSONResponse


class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name


@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something wrong."},
    )
```
""",

    "tutorial/testing.md": """# Testing FastAPI Applications with TestClient

FastAPI includes a `TestClient` based on `httpx` to test your application cleanly with `pytest`.

## Using TestClient

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()


@app.get("/")
async def read_main():
    return {"msg": "Hello World"}


client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}
```

## Testing Endpoints with Authentication and Headers

```python
def test_authenticated_endpoint():
    response = client.get(
        "/users/me", headers={"Authorization": "Bearer fake-token-123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "username" in data
```
""",

    "tutorial/deployment.md": """# Deployment Concepts and Best Practices

Deploying a FastAPI application requires understanding ASGI servers, worker concurrency, and containerization.

## ASGI Servers: Uvicorn and Gunicorn

Uvicorn is an ASGI web server implementation for Python. For production deployments with multiple worker processes:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Alternatively, use Gunicorn managing Uvicorn workers:

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

## Docker Containerization Best Practices

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
"""
}


def download_or_generate_docs(target_dir: str = "data/raw/fastapi_docs"):
    """Download docs from GitHub or write the comprehensive technical documentation corpus."""
    out_path = Path(target_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating technical documentation corpus in '{out_path}'...")
    total_written = 0
    total_bytes = 0

    for relative_path, content in DOCS_CORPUS.items():
        doc_file = out_path / relative_path
        doc_file.parent.mkdir(parents=True, exist_ok=True)
        content_stripped = content.strip() + "\n"
        doc_file.write_text(content_stripped, encoding="utf-8")
        total_written += 1
        total_bytes += len(content_stripped.encode("utf-8"))

    print(f"[OK] Successfully wrote {total_written} markdown documents ({total_bytes / 1024:.1f} KB).")
    return total_written, total_bytes


if __name__ == "__main__":
    download_or_generate_docs()
