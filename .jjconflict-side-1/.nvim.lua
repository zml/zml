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

	function runFixAll()
		local params = vim.lsp.util.make_range_params()
		params.context = {
			only = { "source.fixAll" },
			diagnostics = {},
		}

		-- Send synchronous request (timeout 2 seconds)
		local responses = vim.lsp.buf_request_sync(0, "textDocument/codeAction", params, 1000)
		if not responses then
			return
		end

		for _, resp in pairs(responses) do
			for _, action in ipairs(resp.result or {}) do
				-- If action has edits, apply them
				if action.edit then
					vim.lsp.util.apply_workspace_edit(action.edit, "utf-16")
				end
				-- If the action has a command, execute it
				if action.command then
					vim.lsp.buf.execute_command(action.command)
				end
			end
		end
	end

	vim.api.nvim_create_autocmd("BufWritePre", {
		pattern = { "*.zig", "*.zon" },
		callback = runFixAll,
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
				vim.keymap.set("n", "<Leader>f", function()
					mod.format()
				end, { remap = false, silent = true })
			end,
		})
	end
end
