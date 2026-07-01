-- Get the current working directory (CWD)
local cwd = vim.fn.getcwd()

-- Load the local config file if it exists
local f, err = loadfile(".nvim.local.lua")
if not err then
  local ok, err_f = pcall(f)
  if not ok then
      vim.print(err_f)
  end
else
    vim.lsp.config("zls", {
        cmd = { cwd .. "/tools/zls.sh" },
    })
    vim.lsp.enable("zls")

    vim.api.nvim_create_autocmd('BufWritePre', {
      pattern = { "*.zig", "*.zon" },
      callback = function(ev)
        vim.lsp.buf.code_action({
          context = { only = { "source.fixAll" } },
          apply = true,
        })
      end
    })

    local ok, conform = pcall(require, "conform")

    if ok then
      conform.formatters_by_ft.bzl = { "buildifier" }
      conform.formatters.buildifier = {
        command = cwd .. "/tools/buildifier.sh",
      }

      vim.api.nvim_create_autocmd("FileType", {
          pattern = "bzl",
          callback = function()
              vim.keymap.set('n', '<Leader>f', function()
                  mod.format()
              end, { remap = false, silent = true })
          end
      })
    end
end
