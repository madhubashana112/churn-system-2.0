"""Vector dashboard charts: sharp at print resolution, with no browser dependency."""
from reportlab.platypus import Flowable
from reportlab.lib import colors

PALETTE = ['#98a2b3', '#515ca3', '#e32191', '#f5bc20', '#8055ed', '#0891b2', '#079669']
NAMED = {'CRITICAL':'#b91c1c','HIGH':'#f29100','MEDIUM':'#195bd6','LOW':'#08794e',
         'At risk':'#b91c1c','Healthy':'#059669'}

def chart_color(label, index):
    return colors.HexColor(NAMED.get(label, PALETTE[index % len(PALETTE)]))


def legend_rows(spec):
    if spec.kind == 'doughnut':
        values = spec.datasets[0].values if spec.datasets else []
        total = sum(values)
        return [(label, f'{value:g} ({value / total:.1%})' if total else '0', chart_color(label, i))
                for i, (label, value) in enumerate(zip(spec.labels, values))]
    return [(dataset.label, '', chart_color(dataset.label, i)) for i, dataset in enumerate(spec.datasets)]


class DashboardChart(Flowable):
    def __init__(self, spec, width=505, height=300):
        super().__init__()
        self.spec, self.width, self.height = spec, width, height

    def draw(self):
        c, spec = self.canv, self.spec
        c.saveState()
        c.setFont('ReportText', 8)
        if spec.kind == 'doughnut':
            values = spec.datasets[0].values if spec.datasets else []
            total = sum(values)
            if total:
                angle = 90
                for i, (label, value) in enumerate(zip(spec.labels, values)):
                    if value <= 0:
                        continue
                    sweep = -360 * value / total
                    c.setFillColor(chart_color(label, i))
                    c.setStrokeColor(colors.white)
                    c.wedge(137, 28, 369, 260, angle, sweep, fill=1, stroke=1)
                    angle += sweep
                c.setFillColor(colors.white)
                c.circle(253, 144, 66, fill=1, stroke=0)
                c.setFillColor(colors.HexColor('#242458'))
                c.setFont('ReportBold', 24)
                c.drawCentredString(253, 145, f'{total:g}')
                c.setFont('ReportText', 9)
                c.drawCentredString(253, 124, 'customers')
            else:
                c.drawCentredString(253, 150, 'No data for this selection')
        else:
            left, bottom, width, height = 48, 52, self.width - 64, 210
            scatter = spec.kind == 'scatter'
            if scatter:
                xs = [p.x for d in spec.datasets for p in d.points]
                ys = [p.y for d in spec.datasets for p in d.points]
                xmin, xmax = min([0] + xs), max([1] + xs)
                ymin, ymax = min([0] + ys), max([1] + ys)
            else:
                series = [d.values for d in spec.datasets]
                values = [v for row in series for v in row]
                if spec.stacked:
                    values += [sum(row[i] for row in series if i < len(row)) for i in range(len(spec.labels))]
                ymin, ymax = min([0] + values), max([1] + values) * 1.12
            for tick in range(6):
                y = bottom + tick * height / 5
                c.setStrokeColor(colors.HexColor('#e2e5eb'))
                c.line(left, y, left + width, y)
                c.setFillColor(colors.HexColor('#596477'))
                c.drawRightString(left - 7, y - 3, f'{ymin + (ymax-ymin)*tick/5:.2g}')
            def py(value):
                return bottom + (value-ymin)/(ymax-ymin)*height
            if scatter:
                for tick in range(6):
                    c.drawCentredString(left+width*tick/5, bottom-15, f'{xmin+(xmax-xmin)*tick/5:.2g}')
                for j, dataset in enumerate(spec.datasets):
                    c.setFillColor(chart_color(dataset.label, j))
                    for point in dataset.points:
                        c.circle(left+(point.x-xmin)/(xmax-xmin)*width, py(point.y), 2.6, fill=1, stroke=0)
            else:
                count, groups = max(1, len(spec.labels)), max(1, len(spec.datasets))
                slot = width/count
                for i, label in enumerate(spec.labels):
                    c.setFillColor(colors.HexColor('#596477'))
                    c.setFont('ReportText', 7)
                    # Long category names are keyed to a wrapped table below.
                    display = label if c.stringWidth(label, 'ReportText', 7) < slot-2 else str(i+1)
                    c.drawCentredString(left+(i+.5)*slot, bottom-15, display)
                    base = 0
                    for j, dataset in enumerate(spec.datasets):
                        value = dataset.values[i] if i < len(dataset.values) else 0
                        barwidth = slot*.75/(1 if spec.stacked else groups)
                        x = left+i*slot+slot*.125+(0 if spec.stacked else j*barwidth)
                        start = base if spec.stacked else 0
                        end = start+value
                        c.setFillColor(chart_color(dataset.label,j))
                        c.rect(x, min(py(start),py(end)), max(.5,barwidth-1), abs(py(end)-py(start)),fill=1,stroke=0)
                        if spec.stacked:
                            base=end
            c.setFillColor(colors.HexColor('#596477'))
            c.setFont('ReportText', 8)
            c.drawCentredString(left+width/2, 14, spec.x_label or '')
            c.saveState()
            c.translate(12, bottom+height/2)
            c.rotate(90)
            c.drawCentredString(0,0,spec.y_label or '')
            c.restoreState()
        c.restoreState()
