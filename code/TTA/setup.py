import torch.optim as optim
from TTA.adapt_algorithm import norm, oftta, t3a, tent, pl, shot, sar, tast, tast_bn, dsom, cotta, sotta, lame, note, tsd, tea
import torch
import os
import importlib.util

def setup_dsom(args, model):
    model = dsom.configure_model(model)
    tta_model = dsom.dsom(args, model,
                           steps=1,
                           episodic=True)
    return tta_model

# ---- 按文件路径加载算法模块：规避模块名冲突,供 eata/rotta 这类独立实现使用 ----
def _load_alg(filename):
    p = os.path.join(os.path.dirname(__file__), 'adapt_algorithm', filename)
    spec = importlib.util.spec_from_file_location('alg_' + filename.replace('.py', '').replace('+', '_'), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def setup_source(args, model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    # logger.info(f"model for evaluation: %s", model)
    return model

def setup_norm(args, model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    tta_model = norm.NORM(args, model)
    # logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    # logger.info(f"stats for adaptation: %s", stat_names)
    return tta_model



def setup_pl(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = pl.configure_model(model)
    params, param_names = pl.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = pl.PL(args, model, optimizer,
                           steps=1,
                           episodic=False)

    return tta_model

def setup_shot(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = shot.configure_model(model)
    params, param_names = shot.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = shot.SHOT(args, model, optimizer,
                           steps=1,
                           episodic=False)

    return tta_model


def setup_sar(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    base_optimizer = torch.optim.SGD
    optimizer = sar.SAM(params, base_optimizer, lr= args.lr, momentum=0.9)
    tta_model = sar.SAR(args, model, optimizer,
                           steps=1,
                           episodic=False)

    return tta_model

def setup_t3a(args, model):

    model = t3a.configure_model(model)
    params, param_names = t3a.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = t3a.t3a(args, model, optimizer,
                           steps=1,
                           episodic=False)
    return tta_model

def setup_tast(args, model):

    model = tast.configure_model(model)
    params, param_names = tast.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = tast.TAST(args, model, optimizer,
                           steps=1,
                           episodic=False)
    return tta_model

def setup_tast_bn(args, model):

    model = tast_bn.configure_model(model)
    params, param_names = tast_bn.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = tast_bn.TAST_BN(args, model, optimizer,
                           steps=1,
                           episodic=False)
    return tta_model


def setup_oftta(args, model):
    # 配置OFTTA适配模型 冻结不需要适配的层（如卷积层）激活需要适配的层（如BN层）的训练模式
    model = oftta.configure_model(model)
    # 收集需要优化的参数（通常指BN层参数）（通常指BN层的权重和偏置）
    params, param_names = oftta.collect_params(model)
    # 初始化优化器（实际使用setup_optimizer中的Adam优化器）
    optimizer = setup_optimizer(args, params)
    # 创建OFTTA适配器实例
    tta_model = oftta.OFTTA(args, model, optimizer,
                           steps=1,       # 每个batch执行1次参数更新
                           episodic=False) # 禁用周期性的参数重置

    return tta_model

def setup_tent(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = tent.Tent(args, model, optimizer,
                           steps=1,
                           episodic=False)

    return tta_model
def setup_cotta(args, model):
    """Set up CoTTA adaptation model."""
    model = cotta.configure_model(model)
    params, _ = cotta.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = cotta.CoTTA(args, model, optimizer)
    return tta_model

def setup_note(args, model):
    if args.iabn:
        model = note.convert_iabn(model, args)
    model = note.configure_model(model, args)

    params, _ = note.collect_params(model)
    optimizer = setup_optimizer(args, params)

    note_model = note.NOTE(args, model, optimizer)
    return note_model


def setup_sotta(args, model):
    """
    初始化 SoTTA 流程：
    - 配置模型（只允许 BN / LN / IN 更新）
    - 收集参数
    - 构建优化器（支持 SAM）
    - 返回 SoTTA 实例
    """
    model = sotta.configure_model(model, args)
    params, param_names = sotta.collect_params(model)
    optimizer = setup_optimizer(args, params)

    tta_model = sotta.SoTTA(args, model, optimizer=optimizer,
                      steps=1, episodic=False)

    return tta_model

def setup_lame(args, model):
    """Set up LAME adaptation model.

    Configure the model for evaluation, freeze all parameters,
    initialize affinity and Laplacian label propagation for test-time adaptation.
    """

    model = lame.LAME(args, model)
    return model

def setup_tsd(args, model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tsd.configure_model(model)
    params, param_names = tsd.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = tsd.TSD(args, model, optimizer,
                           steps=1,
                           episodic=False)
    return tta_model


def setup_tea(args, model):

    model = tea.configure_model(model)
    params, param_names = tea.collect_params(model)
    optimizer = setup_optimizer(args, params)
    tta_model = tea.TEA(model, optimizer,
                           episodic=False)
    return tta_model

def setup_eata(args, model):
    m = _load_alg('eata_oftta.py')
    return m.EATA(args, model, steps=1, episodic=True)

def setup_rotta(args, model):
    m = _load_alg('rotta_oftta.py')
    return m.RoTTA(args, model, steps=1, episodic=True)

def setup_optimizer(args, params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if True:
        return optim.Adam(params,
                    lr=args.lr,
                    betas=(0.9, 0.99),
                    weight_decay=5e-4)
    # elif cfg.OPTIM.METHOD == 'SGD':
    #     return optim.SGD(params,
    #                lr=cfg.OPTIM.LR,
    #                momentum=cfg.OPTIM.MOMENTUM,
    #                dampening=cfg.OPTIM.DAMPENING,
    #                weight_decay=cfg.OPTIM.WD,
    #                nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError

def get_adaptation(args, base_model):
    if args.adaption == 'source':
        print('source')
        model = setup_source(args, base_model)
    elif args.adaption == 'norm':
        print('norm')
        model = setup_norm(args, base_model)
    elif args.adaption == "tent":
        print('tent')
        model = setup_tent(args, base_model)
    elif args.adaption == "t3a":
        print('t3a')
        model = setup_t3a(args, base_model)
    elif args.adaption == "tast":
        print('tast')
        model = setup_tast(args, base_model)
    elif args.adaption == "tast_bn":
        print('tast_bn')
        model = setup_tast_bn(args, base_model)
    elif args.adaption == "oftta":
        print('oftta')
        model = setup_oftta(args, base_model)
    elif args.adaption == "dsom":
        print('dsom')
        model = setup_dsom(args, base_model)
    elif args.adaption == "pl":
        print('pl')
        model = setup_pl(args, base_model)
    elif args.adaption == "shot":
        print('shot')
        model = setup_shot(args, base_model)
    elif args.adaption == "sar":
        print('sar')
        model = setup_sar(args, base_model)
    elif args.adaption == "cotta":
        print('cotta')
        model = setup_cotta(args, base_model)
    elif args.adaption == "sotta":
        print('sotta')
        model = setup_sotta(args, base_model)
    elif args.adaption == "lame":
        print('lame')
        model = setup_lame(args, base_model)
    elif args.adaption == "note":
        print('note')
        model = setup_note(args, base_model)
    elif args.adaption == "tsd":
        print('tsd')
        model = setup_tsd(args, base_model)
    elif args.adaption == "tea":
        print('tea')
        model = setup_tea(args, base_model)
    elif args.adaption == "eata":
        print('eata')
        model = setup_eata(args, base_model)
    elif args.adaption == "rotta":
        print('rotta')
        model = setup_rotta(args, base_model)
    else:
        raise ValueError(
            "not exist this adaptation: %r. 请检查 --adaption 是否拼写正确，"
            "或在 TTA/setup.py 的 get_adaptation 中补充该分支。" % (args.adaption,))

    return model