import tensorlayerx as tlx
import paddle
import paddle2tlx
import os
SIZE_UNIT = ['K', 'M', 'G', 'T']
SHM_QUERY_CMD = 'df -h'
SHM_KEY = 'shm'
SHM_DEFAULT_MOUNT = '/dev/shm'


def _parse_size_in_M(size_str):
    if size_str[-1] == 'B':
        num, unit = size_str[:-2], size_str[-2]
    else:
        num, unit = size_str[:-1], size_str[-1]
    assert unit in SIZE_UNIT, 'unknown shm size unit {}'.format(unit)
    return float(num) * 1024 ** (SIZE_UNIT.index(unit) - 1)


def _get_shared_memory_size_in_M():
    try:
        df_infos = os.popen(SHM_QUERY_CMD).readlines()
    except:
        return None
    else:
        shm_infos = []
        for df_info in df_infos:
            info = df_info.strip()
            if info.find(SHM_KEY) >= 0:
                shm_infos.append(info.split())
        if len(shm_infos) == 0:
            return None
        elif len(shm_infos) == 1:
            return _parse_size_in_M(shm_infos[0][3])
        else:
            default_mount_infos = [si for si in shm_infos if si[-1] ==\
                SHM_DEFAULT_MOUNT]
            if default_mount_infos:
                return _parse_size_in_M(default_mount_infos[0][3])
            else:
                return max([_parse_size_in_M(si[3]) for si in shm_infos])
