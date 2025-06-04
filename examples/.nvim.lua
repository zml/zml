-- Get the current working directory (CWD)
local cwd = vim.fn.getcwd()

-- Load the local config file if it exists
local f, err = loadfile(".nvim.local.lua")
if not err then
	f()
else
    local capabilities = vim.lsp.protocol.make_client_capabilities()

	-- Prepend CWD to relative paths
	local zls_cmd = cwd .. "/../zml/tools/zls.sh"

	vim.lsp.config["zls"] = {
        capabilities = capabilities,
		cmd = { zls_cmd },
        root_marker = { "build.zig" },
        filetypes = {"zig"},
	}
    vim.lsp.enable('zls')

    vim.api.nvim_create_autocmd('BufWritePre',{
      pattern = {"*.zig", "*.zon"},
      callback = function(ev)
        vim.lsp.buf.code_action({
          context = { only = { "source.organizeImports" } },
          apply = true,
        })
        vim.lsp.buf.code_action({
          context = { only = { "source.fixAll" } },
          apply = true,
        })
        vim.loop.sleep(100)
      end
    })
end
