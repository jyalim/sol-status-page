Sol Node Status
===============

Scrapes SLURM via `sinfo` and plots node status with `plotly`.

On Sol's `admin.sol.rc.asu.edu`, the user `software` runs the generating shell
and python scripts via `crontab`:

    * * * * * /packages/public/sol-node-status/get-sol-node-status.sh &> /packages/public/sol-node-status/crontab.diag

Live on Open OnDemand page ([Sol status][sol-status], for ASU users).

Shell script will only save every tenth `zstd -19` compressed comma-separated
value (csv) files (every ten minutes). This is determined by an incremented
counter that is stored in `snapshot/.snapshot_modulo.do.not.delete`. When there
are errors, the `zstd` file will not be generated and `csv`s will not be
removed. This is to help determine what occurred in the data and catch all edge
cases.

Open OnDemand
-------------

<div float="center">
 <p align="center"> 
  <img 
    width="95%" 
    src="assets/anim.gif" 
    alt="Animation over April 4, 2023 of the Sol cluster's node utilization." 
  >
 </p>
</div>

Scrapes SLURM via `sinfo` and plots node status with `plotly`.

On Sol's `admin.sol.rc.asu.edu`, the user `software` runs the generating shell
and python scripts via `crontab`:

    * * * * * /packages/public/sol-node-status/get-sol-node-status.sh &> /packages/public/sol-node-status/crontab.diag

Live on Open OnDemand page ([Sol status][sol-status]).

Shell script will only save every tenth `zstd -19` compressed comma-separated
value (csv) files (every ten minutes). This is determined by an incremented
counter that is stored in `snapshot/.snapshot_modulo.do.not.delete`. When there
are errors, the `zstd` file will not be generated and `csv`s will not be
removed. This is to help determine what occurred in the data and catch all edge
cases.

Debugging
---------

Diagnose issues from diagnostics file generated by cron:

    /packages/public/sol-node-status/crontab.diag

and from Python runtime:

    /packages/public/sol-node-status/crontab.diag.log

Note: `crontab.diag` is non-empty upon successful execution with normal
diagnostic statistics on the plotly figure, i.e., 

    len :: total number of nodes being visualized
      M :: number of nodes in figure's horizontal direction
      N :: number of nodes in figure's vertical direction
      w :: figure width in pixels
      h :: figure height in pixels

Planned
-------

* Implement via JS in kibana.
* Automate monthly archiving of snapshot data.

Changes
-------

 Version | Date       | Notes
:-------:|:----------:|:-------------------------------------------------------
 CURRENT | 2023-05-01 | version 0.8
   0.8   | 2023-05-01 | MIG support 
   0.7   | 2023-02-03 | NHC support (escaping quotes in node status REASON)
   0.6   | 2023-01-15 | GRES socket info bug, due to new `sinfo` reporting
   0.5   | 2023-01-06 | GRES length bug fix (`sinfo` fields require spec. width)
   0.4   | 2022-11-23 | GPU Index bug fix
   0.3   | 2022-11-21 | Resized and queue summary statistics added
   0.2   | 2022-11-10 | GPU support
   0.1   | 2022-11-09 | Automated version available on rcstatus.asu.edu
   0.0   | 2022-11-03 | Proof of concept




[sol-status]: https://links.asu.edu/sol-status
