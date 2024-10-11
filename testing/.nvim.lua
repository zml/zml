local f, err = loadfile('.nvim.local.lua')
if not err then
    f()
else 
    require('lspconfig')["zls"].setup {
        cmd = { "tools/zls.sh" },
        settings = {
            zls = {
                enable_autofix = true,
            },
        },
    }
end
