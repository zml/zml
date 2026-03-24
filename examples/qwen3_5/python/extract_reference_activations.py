import torch
import transformers
import zml_utils
from safetensors.torch import load_file, save_file

model_path = "/var/models/Qwen/Qwen3.5-0.8B"
 
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={},
    device="cuda",
)

model, tokenizer = pipeline.model, pipeline.tokenizer

prompt = """Summarize the following text in one sentence:
Le football. Quel sport est plus laid, plus balourd et moins gracieux que le
football ? Quelle harmonie, quelle élégance l'esthète de base pourrait-il bien
découvrir dans les trottinements patauds de vingt-deux handicapés velus qui
poussent des balles comme on pousse un étron, en ahanant des râles vulgaires de
bœufs éteints. Quel bâtard en rut de quel corniaud branlé oserait manifester sa
libido en s'enlaçant frénétiquement comme ils le font par paquets de huit, à
grand coups de pattes grasses et mouillées, en hululant des gutturalités
simiesques à choquer un rocker d'usine
Je vous hais, footballeurs. Vous ne m'avez fait vibrer qu'une fois : le jour où j'ai
appris que vous aviez attrapé la chiasse mexicaine en suçant des frites aztèques.
J'eusse aimé que les amibes vous coupassent les pattes jusqu'à la fin du tournoi.
Mais Dieu n'a pas voulu. Ca ne m'a pas surpris de sa part. Il est des vôtres. Il est
comme vous. Il est partout, tout le temps, quoi qu'on fasse et où qu'on se
planque, on ne peut y échapper."""


pipeline = zml_utils.ActivationCollector(
    pipeline,
    max_layers=1000,
    stop_after_first_step=True,
    include_layer_caches=True,
)
output, activations = pipeline(prompt)
print(output)

filename = "../safetensors/" + model_path.split("/")[-1] + ".activations-compiltest-1.safetensors"
for k in activations.keys():
    if k.endswith("linear_attn.out.1"): # Resize the torch linear conv-state cache which has an extra (useless) dimension
        activations[k] = activations[k][:,:,1:]
activations = {k: v.contiguous() for k, v in activations.items()}

save_file(activations, filename)
print(f"Saved {len(activations)} activations to {filename}")
