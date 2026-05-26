# Optional: let render-group users read Intel PMT telemetry exposed through hwmon.
sudo tee /etc/udev/rules.d/70-intel-pmt-render.rules >/dev/null <<'EOF'
SUBSYSTEM=="intel_pmt", KERNEL=="telem*", RUN+="/bin/chgrp render /sys$devpath/telem", RUN+="/bin/chmod 0440 /sys$devpath/telem"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

# Alternatively, if you don't want to use udev rules, you can manually change the permissions on the telemetry files:

sudo -n chgrp render /sys/class/intel_pmt/telem*/telem
sudo -n chmod 0440 /sys/class/intel_pmt/telem*/telem

lspci | grep Intel

GPU=0000:01:00.0
cat /sys/bus/pci/devices/$GPU/current_link_speed
cat /sys/bus/pci/devices/$GPU/current_link_width
cat /sys/bus/pci/devices/$GPU/max_link_speed
cat /sys/bus/pci/devices/$GPU/max_link_width

# Example workload for fdinfo utilization/process rows.
ONEAPI_DEVICE_SELECTOR=level_zero:0 bazel run //examples/llm     --config=release     --@zml//platforms:cpu=false     --@zml//platforms:oneapi=true     -- --topk=1     --model=/var/models/meta-llama/Llama-3.1-8B-Instruct    --prompt="Tell me a story about a cat in 2 lines"
