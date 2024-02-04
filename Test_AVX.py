import cpuinfo

info = cpuinfo.get_cpu_info()

if 'avx' in info['flags']:
    print("AVX is supported.")
else:
    print("AVX is not supported.")