# Copyright 2024 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Run me with `bazel run //pretty_xla:server -- $PWD`
"""

import json
import logging
import os
import queue
import sys
import tempfile
import time
import traceback
import urllib.parse
from pathlib import Path

import requests
from flask import Response, make_response, request, send_from_directory

import flask
import model_explorer
import requests
from model_explorer.config import ModelExplorerConfig
from model_explorer.extension_manager import ExtensionManager
from model_explorer.utils import convert_adapter_response
from model_explorer.server_directive_dispatcher import ServerDirectiveDispatcher

import hlo2json

ROOT_DIR = "/tmp/zml"
MODEL_EXPLORER_STATIC_FILES = str(Path(model_explorer.__file__).parent / "web_app")
PORT = 19802

app = flask.Flask(__name__)
log = logging.getLogger(__name__)


URL_TEMPLATE = f"/?data=%7B%22models%22:%5B%7B%22url%22:%22__DOT_PATH__%22,%22adapterId%22:%22__EXTENSION__%22%7D%5D,%22nodeData%22:%5B%5D,%22uiState%22:%7B%22paneStates%22:%5B%7B%22deepestExpandedGroupNodeIds%22:%5B%5D,%22selectedNodeId%22:%22%22,%22selectedGraphId%22:%22main%22,%22selectedCollectionLabel%22:%22module_0002.llama.LlamaLM.forward.cpu_after_optimizations.dot%22,%22widthFraction%22:1,%22selected%22:false,%22flattenLayers%22:false%7D%5D%7D%7D&renderer=webgl&show_open_in_new_tab=0&internal_colab=0&ess=0&eetr=0"

server_directive_dispatcher = ServerDirectiveDispatcher()

# Load extensions.
print("Loading extensions...")
extension_manager = model_explorer.extension_manager.ExtensionManager([])
extension_manager.load_extensions()
# Remove tensorflow extensions that create conflict when opening .hlo.pb files.
extension_manager.extensions = [ext for ext in extension_manager.extensions if "pb" not in ext.metadata.fileExts]
extension_manager.extensions.append(model_explorer.extension_manager.RegisteredExtension(
        type="adapter",
        ext_class=hlo2json.XlaDotAdapter,
        metadata=hlo2json.XlaDotAdapter.metadata,
    )
)
extension_manager.extensions.append(model_explorer.extension_manager.RegisteredExtension(
        type="adapter",
        ext_class=hlo2json.HloAdapter,
        metadata=hlo2json.HloAdapter.metadata,
    )
)


extension_metadata_list = extension_manager.get_extensions_metadata()
print(extension_metadata_list)
num_extensions = len(extension_metadata_list)
print(f'Loaded {num_extensions} extension{"" if num_extensions == 1 else "s"}:')
for extension in extension_metadata_list:
    print(f' - {extension["name"]}')


def find_files(base_dir):
    """
    Recursively find all files ending with '.dot' within base_dir.
    Returns a generator of file paths as strings.
    """
    base_path = Path(base_dir)
    for path_obj in base_path.rglob("*.dot"):
        yield str(path_obj)
    for path_obj in base_path.rglob("*.hlo.pb"):
        yield str(path_obj)
    for path_obj in base_path.rglob("*.mlir"):
        yield str(path_obj)


def path_to_url(p):
    if p.endswith(".hlo.pb"):
        extension = "model_explorer_hlo"
    elif p.endswith(".dot"):
        extension = "model_explorer_pretty_xla"
    elif p.endswith(".mlir"):
        extension = "model_explorer_builtin_mlir"
    else:
        extension = "default"

    url = URL_TEMPLATE
    url = url.replace("__DOT_PATH__", urllib.parse.quote(p))
    url = url.replace("__EXTENSION__", extension)
    return url


@app.route("/list")
def list_files():
    ret = f"""
    <html>
      <head><title>ZML Model Explorer Workbench</title></head>
      <body>
        <h1>ZML Model Explorer Workbench</h1>
        <p>Here are links to compiled models:</p>
        <ul>
        """
    for p in sorted(find_files(ROOT_DIR)):
        clean_p = p[len(ROOT_DIR) :].strip("/")
        ret += f'<li><a target="_blank" href="{path_to_url(p)}">{clean_p}</a></li>'
    ret += """
        </ul>
      </body>
    </html>
    """
    return ret


@app.route("/api/v1/check_new_version")
def check_new_version():
    return _make_json_response(
        {
            "version": "",
            "runningVersion": "",
            "releaseUrl": "",
            "desktopAppUrl": "",
        }
    )


@app.route("/api/v1/get_extensions")
def get_extensions():
    """Loads all adapter extensions."""
    return _make_json_response(extension_manager.get_extensions_metadata())


@app.route("/apipost/v1/upload", methods=["POST"])
def upload_file():
    f = request.files["file"]
    file_name = f.filename if f.filename is not None else "no_name"
    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, file_name)
    f.save(file_path)
    return _make_json_response({"path": file_path})


@app.route("/api/v1/send_command")
def send_command():
    cmd_json = json.loads(request.args.get("json", "{}"))
    print("send_command:", cmd_json)
    try:
        resp = extension_manager.run_cmd(cmd_json)
        return _make_json_response(resp)
    except Exception as err:
        traceback.print_exc()
        return _make_json_response({"error": f"{type(err).__name__}: {str(err)}"})
    finally:
        extension_manager.cleanup(cmd_json)


@app.route("/apipost/v1/send_command", methods=["POST"])
def send_command_post():
    try:
        cmd = request.json
        print("send_command_post:", cmd)
        # ext = extension_manager._get_extension_by_id(cmd['extensionId'])
        resp = extension_manager.run_cmd(cmd)
        return _make_json_response(resp)
    except Exception as err:
        traceback.print_exc()
        return _make_json_response({"error": f"{type(err).__name__}: {str(err)}"})
    finally:
        extension_manager.cleanup(request.json)


@app.route("/api/v1/load_graphs_json")
def load_graphs_json():
    if config is None:
        return {}
    graph_index_str = request.args.get("graph_index")
    if graph_index_str is None:
        return {}
    graph_index = int(graph_index_str)
    graphs = config.get_model_explorer_graphs(graph_index)
    if isinstance(graphs, str):
        return _make_json_response(json.loads(graphs))
    else:
        return _make_json_response(convert_adapter_response(graphs))


@app.route("/api/v1/load_node_data")
def load_node_data():
    if config is None:
        return {}
    node_data_index_str = request.args.get("node_data_index")
    if node_data_index_str is None:
        return {}
    node_data_index = int(node_data_index_str)
    node_data = config.get_node_data(node_data_index)
    if isinstance(node_data, str):
        json_str = node_data
    else:
        json_str = node_data.to_json_string()
    return _make_json_response({"content": json_str})


@app.route("/api/v1/read_text_file")
def read_text_file():
    path = request.args.get("path")
    if path is None:
        return _make_json_response({"error": "no file path provided"})
    path = os.path.expanduser(path)

    try:
        with open(path, "r") as file:
            content = file.read()
        return _make_json_response({"content": content})
    except Exception as err:
        return _make_json_response({"error": str(err)})


@app.route("/check_health")
def check_health():
    """Serves check_health request."""
    return "model_explorer_ok"


@app.route("/apipost/v1/update_config", methods=["POST"])
def update_config():
    config_data = request.json
    if config and config_data:
        config.set_transferrable_data(config_data)

        # Ask UI to refresh page with the new url.
        server_directive_dispatcher.broadcast(
            json.dumps(
                {
                    "name": "refreshPage",
                    "url": f"/?data={config.to_url_param_value()}",
                }
            )
        )

    return ""


@app.route("/apistream/server_director")
def server_director_stream():
    def stream():
        directive_queue = server_directive_dispatcher.listen()
        try:
            while True:
                # Try to get a new message.
                try:
                    msg = directive_queue.get(block=False)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    # Ignore if there is no new messages.
                    pass

                # Keep the connection alive.
                yield ": heartbeat\n\n"
                time.sleep(1)
        except:
            # The client closes the connection (i.e. close the browser tab)
            server_directive_dispatcher.remove_listener(directive_queue)

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-cache",
    }
    return flask.Response(
        stream(),
        headers=headers,
        content_type="text/event-stream",
        mimetype="text/event-stream",
    )


@app.route("/")
def send_index_html():
    """Serves index.html."""
    return send_from_directory(MODEL_EXPLORER_STATIC_FILES, "index.html")


@app.route("/<path:path>")
def send_static(path):
    """Serves static files."""
    print("send_static:", path)
    return send_from_directory(MODEL_EXPLORER_STATIC_FILES, path)


def _make_json_response(obj):
    body = json.dumps(obj)
    resp = flask.make_response(body)
    resp.headers["Content-Type"] = "application/json"
    return resp


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ROOT_DIR = sys.argv[1]
        print("Using ROOT_DIR =", ROOT_DIR)

    port = int(sys.argv[2]) if len(sys.argv) > 2 else PORT
    app.run(host="0.0.0.0", port=port)
