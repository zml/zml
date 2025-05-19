# Running Gated Huggingface Models with Token Authentication

Some models have restrictions and may require some sort of approval or agreement
process, which, by consequence, **requires token-authentication with Huggingface**.

The easiest way might be to use the `huggingface-cli login` command.

Alternatively, here is how you can generate a **"read-only public repositories"**
access token to log into your account on Huggingface, directly from `bazel`, in order to download models.

* log in at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
* click on "Create new token"
* give the token a name, eg `zml_public_repos`,
* under _Repositories_, grant the following permission: "Read access to contents of all public gated repos you can access".
* at the bottom click on "Create token".
* copy the token by clicking `Copy`. **You won't be able to see it again.**
* the token looks something like `hf_abCdEfGhijKlM`.
* store the token on your machine (replace the placeholder with your actual token):

You can use the `HUGGINGFACE_TOKEN` environment variable to store the token or use
its standard location:
```
mkdir -p $HOME/.cache/huggingface/; echo <hf_my_token> > "$HOME/.cache/huggingface/token"
```

Now you're ready to download a gated model like `Meta-Llama-3-8b`!

**Example:**

```
# requires token in $HOME/.cache/huggingface/token, as created by the
# `huggingface-cli login` command, or the `HUGGINGFACE_TOKEN` environment variable.
cd examples
bazel run --config=release //llama:Meta-Llama-3-8b
bazel run --config=release //llama:Meta-Llama-3-8b -- --promt="Once upon a time,"
```

