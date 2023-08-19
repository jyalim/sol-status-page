status repo:
  git.latest:
    - name: https://github.com/OSC/osc-systemstatus.git
    - target: /var/www/ood/apps/sys/sol_status
    - user: root
    - unless: /var/www/ood/apps/sys/sol_status

maintain status app.rb:
  file.managed:
    - name: /var/www/ood/apps/sys/sol_status/app.rb
    - source: salt://states/cluster/ood/apps/status/files/app.rb
    - user: root
    - group: root
    - mode: 0644

maintain status layout.erb:
  file.managed:
    - name: /var/www/ood/apps/sys/sol_status/views/layout.erb
    - source: salt://states/cluster/ood/apps/status/files/layout.erb
    - user: root
    - group: root
    - mode: 0644

maintain status sol.html:
  file.symlink:
    - name: /var/www/ood/public/sol.html
    - target: /packages/public/sol-node-status/sol.html
    - force: True

restart sol status webapp by touching restart file:
  file.touch:
    - name: /var/www/ood/apps/sys/sol_status/tmp/restart.txt

run gem bundle install:
  cmd.run:
    - name: bundle install
    - cwd: /var/www/ood/apps/sys/sol_status/
