"""Hexoskin overnight seizure review dashboard."""

__version__ = "1.0.0"

# Display order, top to bottom: cardiac, then the two respiration bands, then
# the accelerometer axes.
CHANNELS = [
    "ECG_I",
    "resp_thorac",
    "resp_abdomi",
    "accel_X",
    "accel_Y",
    "accel_Z",
]

# Times are exported in this timezone. EDF start times are read as UTC and
# converted here, so the switch between EST and EDT is handled from the
# timezone database rather than a fixed offset. Set to "UTC" to disable.
EXPORT_TIMEZONE = "America/Toronto"

# Fixed opening scale for a channel, as the "± value" shown under its name.
# A channel listed here always opens at that scale instead of one measured from
# the recording, which is what you want when rare but enormous excursions would
# otherwise flatten everything else. Reviewers can still zoom freely, and
# "Reset scale" comes back to these values.
#
# Write the number the way the label reads it: "2.2" and "2.2 g" both mean the
# same thing, "220 mg" means 0.22, "50 uV" means 0.00005. Remove a channel from
# this dict to have its scale measured from the data instead.
DEFAULT_HALF_SCALE = {
    "accel_X": "2.2",
    "accel_Y": "2.2",
    "accel_Z": "2.2",
}