import math


def generate_scheduled_sampling_policy(scheduled_sampling_decay_mode, scheduled_sampling_start, scheduled_sampling_end, offset, slope=1.0):
    if scheduled_sampling_decay_mode == "linear":
        return max(min(scheduled_sampling_end, 1.0), scheduled_sampling_start - offset*slope)
    elif scheduled_sampling_decay_mode == "exponential": # scheduled_sampling_start need less than 1.0
        return max(min(scheduled_sampling_end, 1.0), pow(scheduled_sampling_start, offset))
    elif scheduled_sampling_decay_mode == "inverse sigmoid": # scheduled_sampling_start need no less than 1.0
        return max(min(scheduled_sampling_end, 1.0), scheduled_sampling_start / (scheduled_sampling_start + math.exp(offset/scheduled_sampling_start)))
    else:
        return min(scheduled_sampling_end, 1.0)