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
        local bufnr = vim.api.nvim_get_current_buf()
        for _, client in ipairs(vim.lsp.get_clients({ bufnr = bufnr, name = "zls" })) do
            -- Request each action after applying the previous action's edits.
            for _, kind in ipairs({ "source.fixAll", "source.organizeImports" }) do
                local params = vim.lsp.util.make_range_params(0, client.offset_encoding)
                params.context = { only = { kind }, diagnostics = {} }
                local response = client:request_sync("textDocument/codeAction", params, 2000, bufnr)
                if response and not response.err then
                    for _, action in ipairs(response.result or {}) do
                        if action.edit then
                            vim.lsp.util.apply_workspace_edit(action.edit, client.offset_encoding)
                        end
                        if action.command then
                            client:exec_cmd(type(action.command) == "table" and action.command or action,
                                { bufnr = bufnr })
                        end
                    end
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
                    conform.format()
                end, { remap = false, silent = true })
            end,
        })
    end
end
