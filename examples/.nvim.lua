-- Get the current working directory (CWD)
local cwd = vim.fn.getcwd()

-- Load the local config file if it exists
local f, err = loadfile(".nvim.local.lua")
if not err then
	f()
else
	-- Prepend CWD to relative paths
	local zls_cmd = cwd .. "/tools/zls.sh"

	require("lspconfig")["zls"].setup({
		cmd = { zls_cmd },
		settings = {
			zls = {
				enable_autofix = true,
			},
		},
	})
end
