"""LongLive UniPC sampling with clean-prediction guidance."""
import math


def configure_terminal(scheduler, model_scheduler):
    """Change only the fourth update's destination; do not append a UniPC step."""
    if (len(scheduler.timesteps) != 4 or len(scheduler.sigmas) != 5
            or scheduler.config.prediction_type != 'flow_prediction'
            or not scheduler.predict_x0 or scheduler.solver_p is not None
            or scheduler.config.thresholding):
        raise ValueError('Expected a four-step flow UniPC scheduler')
    minimum = float(model_scheduler.sigmas[-1])
    if not math.isfinite(minimum) or not 0 < minimum < float(scheduler.sigmas[-2]):
        raise ValueError('The native model table must supply a positive terminal sigma')
    if float(scheduler.sigmas[-1]) != 0.0:
        raise ValueError('Native UniPC must initially end at exact zero')
    scheduler.sigmas[-1] = minimum
    return minimum


def step_with_clean_hook(scheduler, flow, timestep, sample, hook=None):
    """Guide UniPC's actual clean conversion before its multistep history update.

The raw flow stays a flow. This avoids a lossy clean->BF16-flow round trip and
uses UniPC's continuous sigma rather than the wrapper's nearest-table lookup.
"""
    if hook is None:
        return scheduler.step(flow, timestep, sample, return_dict=False)[0]
    original = scheduler.convert_model_output
    previous = scheduler.__dict__.get('convert_model_output')
    had_override = 'convert_model_output' in scheduler.__dict__
    calls = []

    def converted(model_output, *args, **kwargs):
        clean = original(model_output, *args, **kwargs)
        calls.append(True)
        return hook(clean)

    scheduler.convert_model_output = converted
    try:
        result = scheduler.step(flow, timestep, sample, return_dict=False)[0]
        if len(calls) != 1:
            raise RuntimeError('UniPC clean-conversion contract changed')
        return result
    finally:
        if had_override:
            scheduler.convert_model_output = previous
        else:
            del scheduler.convert_model_output


def sample_block(generator, scheduler, noisy, conditioning, caches, current_start,
                 *, terminal=False, clean_hook=None, phase_hook=None):
    """Apply native UniPC updates and the separate terminal clean prediction."""
    latents = noisy
    for step, timestep in enumerate(scheduler.timesteps):
        if phase_hook:
            phase_hook('denoise', step)
        flow, _ = generator(latents, timestep, conditioning, caches, current_start)
        hook = None if clean_hook is None else lambda clean, k=step: clean_hook(clean, k)
        latents = step_with_clean_hook(scheduler, flow, timestep, latents, hook)
    if terminal:
        if phase_hook:
            phase_hook('terminal_prediction', 4)
        _, clean = generator(latents, 0, conditioning, caches, current_start)
        latents = clean if clean_hook is None else clean_hook(clean, 4)
    return latents
