# CAD export checklist (SolidWorks / Onshape / other)

Goal: fill in `vehicle/vehicle_template.json` (copy it to `vehicle/my_rocket.json`) and load it
with `tvc.vehicle_io.load_vehicle`. **SI units only**: set the CAD document units to MMGS or
convert (m, kg, kg*m^2, N, s). Unit slips (mm vs m, g vs kg) are the most likely error; the
loader rejects obviously wrong values.

## Frame (the one thing that must be right)
- Pick the **gimbal pivot** (the point the thrust vector swings about) as the reference.
- All `x_*` values are distance **forward toward the nose** from that pivot, along the rocket axis.
- Inertias are about the **center of mass**, not about the origin. In SolidWorks use
  *Mass Properties > "Moments of inertia, taken at the center of mass and aligned with the output
  coordinate system"* with a coordinate system whose X axis is the rocket axis (nose = +X).
  In Onshape: *Measure > Mass properties*, with a mate connector on the axis.
- Ixx = roll (about the rocket axis), Iyy = lateral (Izz should be ~equal; if not, tell me).

## Two configurations to export
1. **Ignition**: motor installed with full propellant. Read: total mass, CoM x, Ixx, Iyy.
2. **Burnout**: same assembly with the propellant mass removed (or the motor swapped for its
   empty casing). Read the same four numbers.
   The loader checks `ignition mass - burnout mass` against the motor's propellant mass (10%).

## Geometry
- Overall length, body radius (the largest radius used for drag reference area).
- Gimbal pivot location along the body (this fixes the moment arm `d` = CoM - pivot).
- Nozzle position relative to the rail buttons (`exit_offset_m`) and rail length.

## Aerodynamics (CAD alone can't give these; best-effort)
- **Center of pressure** x from the pivot: use a Barrowman calculation (OpenRocket/RockSim) on
  the same geometry, or SolidWorks Flow Simulation if available.
- `cn_alpha` (per rad) from the same tool; `cd_axial` from OpenRocket or a CFD/estimate.
- If skipped, the placeholder values stay and controller tuning is only approximate.

## TVC mount / actuators
- Max gimbal deflection per axis (deg), from mount geometry or servo-horn limits.
- Servo rate (deg/s, from the datasheet at your voltage), lag time constant and command delay.

## Motor
- Thrust curve (time, thrust) from thrustcurve.org or your test stand; propellant mass.
