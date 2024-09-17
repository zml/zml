
# Huggingface Token Authentication

Some models have restrictions and may require some sort of approval or
agreement process, which, by consequence, **requires token-authentication with
Huggingface**.

Here is how you can generate a **"read-only public repositories"** access token
to log into your account on Huggingface, directly from `bazel`, in order to
download models.

* log in at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
* click on "Create new token"
* give the token a name, eg `zml_public_repos`
* under _Repositories_, grant the following permission: "Read access to
  contents of all public gated repos you can access".
* at the bottom, click on "Create token".
* copy the token by clicking `Copy`. **You won't be able to see it again.**
* the token looks something like `hf_abCdEfGhijKlM`.
* store the token on your machine (replace the placeholder with your actual
  token):

```
echo -n <hf_my_token> > `$HOME/.cache/huggingface/token`
```

The `-n` is important in order to not append an "end of line" character at the
end of the file that would corrupt the token.

Now you're ready to download a gated model like `Meta-Llama-3-8b`!

**Example:**

```
# requires token in $HOME/.cache/huggingface/token
cd examples
bazel run -c opt //llama:Meta-Llama-3-8b
bazel run -c opt //llama:Meta-Llama-3-8b -- --promt="Once upon a time,"
```


