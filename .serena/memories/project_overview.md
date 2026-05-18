# Hive — Project Overview

## Purpose
Multi-agent SDLC framework ("Collective Intelligence Building Software"). Assembles a team of AI agents that collaborate through a 13-phase software development lifecycle to turn a feature request into a complete, tested, documented, shippable project — with checkpoints, memory, and human oversight.

## Agent Crew
- Scout — researches problem space
- Penny — interviews user, writes PRD + Handover.md
- Archie — designs architecture
- Judge — ratifies the plan
- Dev team — builds all source files (parallel, by dependency layer)
- Quinn — quality review + SIT test plan
- Alex — writes UAT scenarios as named pseudo-user
- Morgan — final delivery checklist

## Tech Stack
- Python 3.12+ (also supports 3.13, 3.14)
- httpx (only runtime dep)
- anthropic SDK (optional extra)
- pytest + ruff (dev)
- hatchling build backend
- Package name: `hive-ept`, CLI: `hive`

## Output
`projects/{project_name}/` containing `src/`, `docs/`, `memory/`, `checkpoints/`

## Remotes
- GitHub public: `https://github.com/naveenkumarbaskaran/hive`
- (check local git remote for tools.sap remote)
