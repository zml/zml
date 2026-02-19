# Chat Data Fixtures

These files are reusable **input data fixtures** for chat-template rendering.

- Each file is a JSON object that can be passed as the template context.
- The intent is to avoid repeating large `messages`/`tools` payloads in every generated test case.
- Later, generate concrete test cases as a Cartesian product of:
  - `chat/templates/*.jinja`
  - `chat/data/*.json`

Suggested generated test-case shape (filename references only):

```json
{
  "name": "<template-stem>__<data-stem>",
  "template_file": "chat/templates/<template-file>",
  "data_file": "chat/data/<data-file>"
}
```
