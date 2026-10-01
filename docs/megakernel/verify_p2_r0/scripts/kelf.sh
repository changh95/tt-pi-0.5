#!/bin/bash
# md5 of the whole_* / mk_* ELF loadable content (/home/deepgadget/experiments/gr00t/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy -O binary) in a cache dir; argv1 = cache, argv2 = label
echo "# $2 $(date +%F_%T) $1"
for f in $(find $1 -path '*/kernels/*' \( -name 'brisc.elf' -o -name 'ncrisc.elf' -o -name 'trisc?.elf' \) | sort); do
  k=$(echo $f | sed 's#.*/kernels/##')
  case $k in whole_*|mk_*) /home/deepgadget/experiments/gr00t/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy -O binary $f /tmp/claude-1000/kelf_$$.bin && echo "$(md5sum < /tmp/claude-1000/kelf_$$.bin | cut -c1-32) $(stat -c %s /tmp/claude-1000/kelf_$$.bin) $k";; esac
done
rm -f /tmp/claude-1000/kelf_$$.bin
