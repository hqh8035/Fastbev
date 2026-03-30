from .config_fastbev_dinov3 import get_cfg_defaults as get_cfg_defaults_fastbev_dinov3
from .config_fastbev_v1 import get_cfg_defaults as get_cfg_defaults_fastbev_v1
from .config_fastbev_v2 import get_cfg_defaults as get_cfg_defaults_fastbev_v2
from .config_fastbev_v3 import get_cfg_defaults as get_cfg_defaults_fastbev_v3
from .config_fastbev_v4 import get_cfg_defaults as get_cfg_defaults_fastbev_v4
from .config_fastbev_v5 import get_cfg_defaults as get_cfg_defaults_fastbev_v5
from .config_fastbev_v6 import get_cfg_defaults as get_cfg_defaults_fastbev_v6
from .config_fastbev_v7 import get_cfg_defaults as get_cfg_defaults_fastbev_v7


__all__ = ['get_cfg_defaults_fastbev_dinov3',
           'get_cfg_defaults_fastbev_v1',
           'get_cfg_defaults_fastbev_v2',
           'get_cfg_defaults_fastbev_v3',
           'get_cfg_defaults_fastbev_v4',
           'get_cfg_defaults_fastbev_v5',
           'get_cfg_defaults_fastbev_v6',
           'get_cfg_defaults_fastbev_v7']


def get_cfg_defaults(model_name):
    if model_name == 'fastbev_dinov3':
        return get_cfg_defaults_fastbev_dinov3()
    elif model_name == 'fastbev_v1':
        return get_cfg_defaults_fastbev_v1()
    elif model_name == 'fastbev_v2':
        return get_cfg_defaults_fastbev_v2()
    elif model_name == 'fastbev_v3':
        return get_cfg_defaults_fastbev_v3()
    elif model_name == 'fastbev_v4':
        return get_cfg_defaults_fastbev_v4()
    elif model_name == 'fastbev_v5':
        return get_cfg_defaults_fastbev_v5()
    elif model_name == 'fastbev_v6':
        return get_cfg_defaults_fastbev_v6()
    elif model_name == 'fastbev_v7':
        return get_cfg_defaults_fastbev_v7()
    else:
        raise ValueError(f"Invalid model name: {model_name}") 