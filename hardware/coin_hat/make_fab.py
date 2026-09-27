#!/usr/bin/env python3
"""Generate fab and assembly files for JLCPCB and PCBWay from the KiCad project.

Run from anywhere (needs kicad-cli on PATH):

    python3 hardware/coin_hat/make_fab.py

Writes hardware/coin_hat/fab/:
    coin_hat_gerbers.zip            Gerbers + Excellon drill (both fabs)
    jlcpcb/coin_hat_bom_jlcpcb.csv  SMD parts JLCPCB assembles (LCSC part numbers)
    jlcpcb/coin_hat_cpl_jlcpcb.csv  pick-and-place for those parts
    pcbway/coin_hat_bom_pcbway.csv  every part, THT lines marked customer-soldered
    pcbway/coin_hat_centroid_pcbway.csv

Part numbers come from the LCSC / Manufacturer / MPN fields on the schematic
symbols, so edit them there and re-run.
"""

import csv
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCH = HERE / 'coin_hat.kicad_sch'
PCB = HERE / 'coin_hat.kicad_pcb'
OUT = HERE / 'fab'

GERBER_LAYERS = 'F.Cu,B.Cu,F.Paste,B.Paste,F.Silkscreen,B.Silkscreen,F.Mask,B.Mask,Edge.Cuts'

# SMD parts the fab places. U1 (Pico, castellated module) and the solder jumpers
# are SMD footprints but not assembled.
NOT_ASSEMBLED = re.compile(r'^(U1|JP\d+)$')

# JLCPCB's parts library orients some packages differently from KiCad's.
# Offsets (degrees, added to KiCad's rotation) follow the widely used
# kicad-jlcpcb-tools table. Always check JLCPCB's placement preview.
JLC_ROTATION = [(r'^SOT-23', 180), (r'^SOT-353', 180), (r'^SOIC-', 270)]


def run(*args):
    subprocess.run(args, check=True, capture_output=True, text=True)


def read_components(netlist):
    """ref -> {value, footprint, fields...} for every BOM component."""
    s = netlist.read_text()
    comps = {}
    for m in re.finditer(r'\(comp \(ref "([^"]+)"\)(.*?)\(tstamps', s, re.S):
        body = m[2]
        c = {'value': re.search(r'\(value "([^"]*)"\)', body)[1],
             'footprint': re.search(r'\(footprint "([^"]*)"\)', body)[1]}
        c.update(re.findall(r'\(field \(name "([^"]+)"\) "([^"]*)"\)', body))
        if '(property (name "exclude_from_bom")' in body:
            continue
        comps[m[1]] = c
    return comps


def natural(ref):
    m = re.match(r'([A-Z]+)(\d+)', ref)
    return (m[1], int(m[2])) if m else (ref, 0)


def group(comps, key):
    rows = defaultdict(list)
    for ref, c in comps.items():
        rows[key(c)].append(ref)
    return sorted(((k, sorted(v, key=natural)) for k, v in rows.items()), key=lambda kv: natural(kv[1][0]))


def placements(tmp):
    pos = tmp / 'pos.csv'
    run('kicad-cli', 'pcb', 'export', 'pos', '--format', 'csv', '--units', 'mm', '--side', 'both',
        '--smd-only', '-o', str(pos), str(PCB))
    with pos.open() as f:
        return {r['Ref']: r for r in csv.DictReader(f)}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'jlcpcb').mkdir(parents=True)
    (OUT / 'pcbway').mkdir()
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)

        gerb = tmp / 'gerbers'
        gerb.mkdir()
        run('kicad-cli', 'pcb', 'export', 'gerbers', '--layers', GERBER_LAYERS, '--subtract-soldermask',
            '-o', str(gerb), str(PCB))
        run('kicad-cli', 'pcb', 'export', 'drill', '--format', 'excellon', '--drill-origin', 'absolute',
            '--excellon-units', 'mm', '--excellon-separate-th',
            '-o', str(gerb) + '/', str(PCB))
        with zipfile.ZipFile(OUT / 'coin_hat_gerbers.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(gerb.iterdir()):
                z.write(f, f.name)

        net = tmp / 'coin_hat.net'
        run('kicad-cli', 'sch', 'export', 'netlist', '--format', 'kicadsexpr', '-o', str(net), str(SCH))
        comps = read_components(net)
        pos = placements(tmp)

    smd = {r: c for r, c in comps.items() if r in pos and not NOT_ASSEMBLED.match(r)}
    missing = [r for r, c in smd.items() if not c.get('LCSC')]
    if missing:
        raise SystemExit(f'SMD parts without an LCSC number: {missing}')

    # --- JLCPCB -----------------------------------------------------------
    with (OUT / 'jlcpcb' / 'coin_hat_bom_jlcpcb.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Comment', 'Designator', 'Footprint', 'LCSC Part #'])
        for (val, fp, lcsc), refs in group(smd, lambda c: (c['value'], c['footprint'].split(':')[1], c['LCSC'])):
            w.writerow([val, ','.join(refs), fp, lcsc])
    with (OUT / 'jlcpcb' / 'coin_hat_cpl_jlcpcb.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Designator', 'Mid X', 'Mid Y', 'Layer', 'Rotation'])
        for ref in sorted(smd, key=natural):
            p = pos[ref]
            rot = float(p['Rot'])
            for pat, off in JLC_ROTATION:
                if re.match(pat, p['Package']):
                    rot += off
            w.writerow([ref, f"{float(p['PosX']):.4f}mm", f"{float(p['PosY']):.4f}mm",
                        'Top' if p['Side'] == 'top' else 'Bottom', f'{rot % 360:g}'])

    # --- PCBWay -----------------------------------------------------------
    with (OUT / 'pcbway' / 'coin_hat_bom_pcbway.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Item #', 'Designator', 'Qty', 'Manufacturer', 'Mfg Part #', 'Description / Value',
                    'Package/Footprint', 'Type', 'Your Instructions / Notes'])
        key = lambda c: (c['value'], c['footprint'].split(':')[1], c.get('Manufacturer', ''), c.get('MPN', ''),
                         c.get('LCSC', ''))
        for i, ((val, fp, mfr, mpn, lcsc), refs) in enumerate(group(comps, key), 1):
            assembled = all(r in smd for r in refs)
            note = 'Assemble' if assembled else 'Do not assemble - customer solders'
            if lcsc:
                note += f' (LCSC {lcsc})'
            w.writerow([i, ','.join(refs), len(refs), mfr, mpn, val, fp, 'SMD' if assembled else 'THT/module', note])
    with (OUT / 'pcbway' / 'coin_hat_centroid_pcbway.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Designator', 'Value', 'Package', 'Mid X (mm)', 'Mid Y (mm)', 'Rotation', 'Layer'])
        for ref in sorted(smd, key=natural):
            p = pos[ref]
            w.writerow([ref, p['Val'], p['Package'], p['PosX'], p['PosY'], p['Rot'],
                        'Top' if p['Side'] == 'top' else 'Bottom'])

    print(f'{len(smd)} SMD parts assembled, {len(comps)} BOM parts total -> {OUT}')


if __name__ == '__main__':
    main()
