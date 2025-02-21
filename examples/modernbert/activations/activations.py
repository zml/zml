import logging
import torch
from transformers import pipeline
from tools.zml_utils import ActivationCollector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

MODEL_NAME: str = "answerdotai/ModernBERT-base"


def main() -> None:
    try:
        log.info("Start running main()")

        log.info(f"CPU capability : `{torch.backends.cpu.get_cpu_capability()}`")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info(f"Loading model : `{MODEL_NAME}`")

        fill_mask_pipeline = pipeline(
            "fill-mask",
            model=MODEL_NAME,
            device_map=device,
        )
        model, tokenizer = fill_mask_pipeline.model, fill_mask_pipeline.tokenizer
        log.info(
            f"Model loaded successfully {model.config.architectures} - `{model.config.torch_dtype}` - {tokenizer.model_max_length} max tokens"  # noqa: E501
        )

        # Wrap the pipeline, and extract activations.
        # Activations files can be huge for big models,
        # so let's stop collecting after 1000 layers.
        zml_pipeline = ActivationCollector(
            fill_mask_pipeline, max_layers=1000, stop_after_first_step=True
        )

        input_text = "Paris is the [MASK] of France."
        outputs, activations = zml_pipeline(input_text)
        log.info(f"ouputs : {outputs}")

        filename = MODEL_NAME.split("/")[-1] + ".activations.pt"
        torch.save(activations, filename)
        log.info(f"Saved {len(activations)} activations to {filename}")

        log.info("End running main()")
    except Exception as exception:
        log.error(exception)
        raise


if __name__ == "__main__":
    main()
