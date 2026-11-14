# Break the Agent: 2D Structural Design

*Built for **Junction 2026** to solve **AfterQuery’s “Break the Agent” challenge**: creating an RL environment that exposes real AI failure modes with exploit-resistant rewards.*

**Today’s AI agents can write thousands of lines of code in seconds. Ask them to design a real physical part—and they fail basic physics.**

In aerospace and automotive engineering, removing excess weight without sacrificing strength is everything. Most AI benchmarks only test text or 2D pixel grids that you could never manufacture. We built a **(RL) environment for real 2D structural design**: agents carve shapes out of a solid metal plate and get millisecond physics feedback on whether the part will hold up under load, scored against theoretical **topological optimization ($C^*$)**.

- **Real Geometry**: Agents cut holes, slots, and rounded pockets—not raw pixel grids—exportable to standard manufacturing files
- **Millisecond Physics**: Instant structural simulation measuring stiffness, weight, and peak stress
- **Harbor + RL Suite**: 3 graded multi-load benchmarks with an isolated anti-cheat verifier

![CAD modeling is a laborous task.](assets/assemblies.png)
