"""Static source and documentation inspection."""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

from . import __version__
from .phrases import FORECAST, NOFORECAST

GENERIC = set('forecast forecast_time_series forecast_series make_forecast make_forecasts predict_future predict_future_values forecast_future forecast_next predict_next_values generate_forecast generate_forecasts create_forecast run_forecast get_forecast get_forecasts compute_forecast produce_forecast build_forecast forecast_values forecast_demand forecast_sales forecast_prices forecast_trend'.split())
ROUTES = (
    ('statsmodels.tsa', set('ARIMA SARIMAX ExponentialSmoothing SimpleExpSmoothing Holt'.split()), {'forecast', 'predict', 'get_prediction'}),
    ('prophet', {'Prophet'}, {'predict'}),
    ('pmdarima', {'auto_arima'}, {'predict', 'forecast'}),
    ('neuralforecast', {'NeuralForecast'}, {'predict'}),
    ('darts', set('ARIMA ExponentialSmoothing Prophet NBEATSModel TFTModel TCNModel RNNModel TransformerModel AutoARIMA'.split()), {'predict'}),
    ('sktime', set('ARIMA ExponentialSmoothing ThetaForecaster NaiveForecaster AutoARIMA'.split()), {'predict'}),
)


def normalize(text):
    return ' '.join(text.lower().replace("n't", ' not').replace('-', ' ').split())


def claims(text):
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
            lines.append('')
        elif match:
            fence = match[1]
            lines.append('')
        else:
            lines.append(line)
    positive = negative = False
    for paragraph in re.split(r'\n\s*\n', '\n'.join(lines)):
        paragraph = normalize(paragraph)
        for phrase in sorted(NOFORECAST, key=lambda x: (-len(x), x)):
            paragraph, count = re.subn(r'\b' + re.escape(phrase) + r'\b', ' ', paragraph)
            negative |= bool(count)
        positive |= any(re.search(r'\b' + re.escape(x) + r'\b', paragraph) for x in sorted(FORECAST, key=lambda x: (-len(x), x)))
    return positive, negative


def evidence(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    nodes = list(ast.walk(tree))
    aliases = {}
    imports = set()
    shadows = {x.name for x in nodes if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    bindings = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for item in node.names:
                imports.add(item.name)
                aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            imports.add(module)
            for item in node.names:
                aliases[item.asname or item.name] = module + '.' + item.name
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bindings[node.target.id] = node.value

    def resolved(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return resolved(node.value) + '.' + node.attr
        return ''

    def shadowed(node):
        if isinstance(node, ast.Name):
            return node.id in shadows or resolved(node).split('.')[-1] in shadows
        if isinstance(node, ast.Attribute):
            return node.attr in shadows or shadowed(node.value)
        return False

    def construction(node, route, seen=None, fitted=False):
        seen = set() if seen is None else seen
        if isinstance(node, ast.Name) and node.id in bindings and node.id not in seen:
            return construction(bindings[node.id], route, seen | {node.id}, fitted)
        if not isinstance(node, ast.Call):
            return False
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'fit' and not fitted:
            return construction(node.func.value, route, seen, True)
        module, names, _ = route
        return (isinstance(node.func, (ast.Name, ast.Attribute))
                and resolved(node.func).split('.')[-1] in names
                and not shadowed(node.func)
                and any(x == module or x.startswith(module + '.') or
                        (module == 'statsmodels.tsa' and x == 'statsmodels' and resolved(node.func).startswith('statsmodels.tsa.')) for x in imports))

    for node in nodes:
        if not isinstance(node, ast.Call) or shadowed(node.func):
            continue
        name = resolved(node.func).split('.')[-1]
        if name in GENERIC and (isinstance(node.func, ast.Attribute) or
                                isinstance(node.func, ast.Name) and node.func.id in aliases):
            return True
        if isinstance(node.func, ast.Attribute):
            for route in ROUTES:
                base = node.func.value
                allowed = isinstance(base, ast.Name) or (isinstance(base, ast.Call) and isinstance(base.func, ast.Attribute) and base.func.attr == 'fit')
                if isinstance(base, ast.Name) and route[0] != 'statsmodels.tsa':
                    bound = bindings.get(base.id)
                    if isinstance(bound, ast.Call) and isinstance(bound.func, ast.Attribute) and bound.func.attr == 'fit':
                        allowed = False
                if allowed and node.func.attr in route[2] and construction(base, route):
                    return True
    return False


def audit(root, ignore=()):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise OSError('root must be a readable directory')
    positive = negative = False
    production, tests = [], []
    # Explicit traversal ensures symlink directories are never followed.
    def walk(directory):
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(root)
            if path.is_symlink() or any(relative.as_posix().startswith(x) for x in ignore):
                continue
            if path.is_dir():
                yield from walk(path)
            elif path.is_file():
                yield path, relative
    for path, relative in walk(root):
        parts = relative.parts
        if relative.as_posix() == 'README.md' or path.suffix == '.md' and 'docs' in parts[:-1]:
            yes, no = claims(path.read_text(encoding='utf-8'))
            positive |= yes
            negative |= no
        if path.suffix != '.py' or {'fixtures', 'docs', 'examples'} & set(parts[:-1]):
            continue
        if evidence(path.read_text(encoding='utf-8')):
            target = tests if 'tests' in parts[:-1] or path.name.startswith('test_') or path.name.endswith('_test.py') else production
            target.append(relative.as_posix())
    findings = []
    if positive and not production:
        findings.append({'rule': 'documented-time-series-forecasting-claim-test-only' if tests else 'documented-time-series-forecasting-claim-false',
                         'severity': 'MEDIUM' if tests else 'HIGH',
                         'message': 'Documentation claim has evidence only in tests.' if tests else 'Documentation claim has no production evidence.'})
    if negative and production:
        findings.append({'rule': 'documented-no-time-series-forecasting-claim-false', 'severity': 'HIGH', 'message': 'Documentation denial conflicts with production evidence.'})
    return {'verdict': 'FAIL' if findings else 'PASS', 'findings': findings,
            'production_evidence': production, 'test_evidence': tests}


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, 'error: ' + message + '\n')


def main(argv=None):
    parser = Parser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True, parser_class=Parser)
    command = commands.add_parser('audit')
    command.add_argument('repo', nargs='?', default='.')
    command.add_argument('--json', action='store_true')
    command.add_argument('--verbose', action='store_true')
    command.add_argument('--ignore', action='append', default=[])
    command.add_argument('--fail-on', type=str.lower, choices=['high', 'medium'], default='high')
    command.add_argument('--version', action='version', version=__version__)
    args = parser.parse_args(argv)
    try:
        result = audit(args.repo, args.ignore)
    except (OSError, UnicodeError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result['verdict'])
        for finding in result['findings']:
            print('{severity} {rule}: {message}'.format(**finding))
        if args.verbose:
            print('Production evidence: ' + ', '.join(result['production_evidence']))
            print('Test evidence: ' + ', '.join(result['test_evidence']))
    return 2 if any(x['severity'] == 'HIGH' or args.fail_on == 'medium' for x in result['findings']) else 0
