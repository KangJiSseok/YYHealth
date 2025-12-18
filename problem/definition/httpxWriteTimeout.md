## 📌 이슈: Qdrant upsert 중 HTTP WriteTimeout 발생
### 🔴 문제 상황

PDF 문서를 chunking 및 embedding한 뒤,
생성된 벡터 데이터를 Qdrant에 적재(upsert)하는 과정에서 다음과 같은 에러가 발생했다.

httpcore.WriteTimeout: timed out
qdrant_client.http.exceptions.ResponseHandlingException: timed out


에러는 임베딩이 모두 완료된 이후,
Qdrant에 points를 업로드하는 단계에서 발생하였다.

## 🔍 원인 분석

초기 구현에서는 전체 chunk(1,672개)를 단일 요청으로 upsert하는 방식이었다.

벡터 수: 1,672개

벡터 차원: 3,072

payload에 원문 텍스트 포함

결과적으로 HTTP 요청 body 크기가 과도하게 커짐

이로 인해:

Qdrant 서버가 요청을 처리하는 동안

클라이언트(httpx) 측에서 WriteTimeout이 발생

서버 장애가 아닌 대용량 단일 요청에 따른 네트워크 타임아웃 문제로 판단

## 🛠 해결 전략

대량 데이터를 한 번에 전송하는 방식 대신,
batch 단위로 나누어 upsert하는 구조로 개선하였다.

적용한 변경 사항

Upsert batch 처리

한 번에 100~200 points씩 분할 전송

Qdrant client timeout 증가

대량 처리 환경을 고려해 timeout을 60초로 설정

진행 로그 추가

batch 단위로 업로드 진행 상황을 확인할 수 있도록 로그 출력

# ✅ 개선 결과

HTTP WriteTimeout 에러 재발 ❌

모든 chunk(1,672개)가 안정적으로 Qdrant에 적재됨

Qdrant Dashboard에서 points_count 정상 확인

대용량 문서 처리 시에도 확장 가능한 구조 확보

```aiignore
/Users/kangjiseok/Desktop/YYHealth/.venv/bin/python /Users/kangjiseok/Desktop/YYHealth/app/run_embed_upsert.py 
[embed_upsert] Loaded 1672 chunks from /Users/kangjiseok/Desktop/YYHealth/data/chunks.jsonl
Traceback (most recent call last):
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpx/_transports/default.py", line 72, in map_httpcore_exceptions
    yield
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpx/_transports/default.py", line 236, in handle_request
    resp = self._pool.handle_request(req)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpcore/_sync/connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpcore/_sync/http11.py", line 136, in handle_request
    raise exc
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpcore/_sync/http11.py", line 88, in handle_request
    self._send_request_body(**kwargs)
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpcore/_sync/http11.py", line 159, in _send_request_body
    self._send_event(event, timeout=timeout)
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpcore/_sync/http11.py", line 166, in _send_event
    self._network_stream.write(bytes_to_send, timeout=timeout)
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpcore/_backends/sync.py", line 135, in write
    with map_exceptions(exc_map):
  File "/opt/homebrew/Cellar/python@3.11/3.11.14_1/Frameworks/Python.framework/Versions/3.11/lib/python3.11/contextlib.py", line 158, in __exit__
    self.gen.throw(typ, value, traceback)
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpcore/_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
httpcore.WriteTimeout: timed out

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/qdrant_client/http/api_client.py", line 106, in send_inner
    response = self._client.send(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpx/_client.py", line 926, in send
    response = self._send_handling_auth(
               ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpx/_client.py", line 954, in _send_handling_auth
    response = self._send_handling_redirects(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpx/_client.py", line 991, in _send_handling_redirects
    response = self._send_single_request(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpx/_client.py", line 1027, in _send_single_request
    response = transport.handle_request(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpx/_transports/default.py", line 235, in handle_request
    with map_httpcore_exceptions():
  File "/opt/homebrew/Cellar/python@3.11/3.11.14_1/Frameworks/Python.framework/Versions/3.11/lib/python3.11/contextlib.py", line 158, in __exit__
    self.gen.throw(typ, value, traceback)
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/httpx/_transports/default.py", line 89, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.WriteTimeout: timed out

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/kangjiseok/Desktop/YYHealth/app/run_embed_upsert.py", line 93, in <module>
    run()
  File "/Users/kangjiseok/Desktop/YYHealth/app/run_embed_upsert.py", line 88, in run
    upsert_chunks(chunks)
  File "/Users/kangjiseok/Desktop/YYHealth/app/run_embed_upsert.py", line 71, in upsert_chunks
    client.upsert(
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/qdrant_client/qdrant_client.py", line 1364, in upsert
    return self._client.upsert(
           ^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/qdrant_client/qdrant_remote.py", line 1766, in upsert
    http_result = self.openapi_client.points_api.upsert_points(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/qdrant_client/http/api/points_api.py", line 1667, in upsert_points
    return self._build_for_upsert_points(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/qdrant_client/http/api/points_api.py", line 852, in _build_for_upsert_points
    return self.api_client.request(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/qdrant_client/http/api_client.py", line 79, in request
    return self.send(request, type_)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/qdrant_client/http/api_client.py", line 96, in send
    response = self.middleware(request, self.send_inner)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/qdrant_client/http/api_client.py", line 205, in __call__
    return call_next(request)
           ^^^^^^^^^^^^^^^^^^
  File "/Users/kangjiseok/Desktop/YYHealth/.venv/lib/python3.11/site-packages/qdrant_client/http/api_client.py", line 108, in send_inner
    raise ResponseHandlingException(e)
qdrant_client.http.exceptions.ResponseHandlingException: timed out

종료 코드 1(으)로 완료된 프로세스

```
