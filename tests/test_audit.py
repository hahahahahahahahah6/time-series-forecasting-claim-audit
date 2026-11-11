import ast
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from time_series_forecasting_claim_audit import phrases
from time_series_forecasting_claim_audit.audit import audit, claims, evidence, main, GENERIC, ROUTES

EXPECTED_FORECAST = ('time series forecasting', 'time series forecast', 'time series forecasts', 'forecasts time series', 'forecasting time series', 'forecast time series', 'time series prediction', 'time series predictions', 'predicts time series', 'predicting time series', 'forecast future values', 'forecasts future values', 'predict future values', 'predicts future values', 'future value prediction', 'future values prediction', 'temporal forecasting', 'time series modeling', 'time series models', 'forecasts the future', 'forecast the future', 'predicts the future', 'predicting the future', 'future forecasting', 'multi step forecasting', 'multi horizon forecasting', 'long term forecasting', 'short term forecasting', 'one step ahead prediction', 'multi step prediction', 'next value prediction', 'predicts next values', 'forecasts next values', 'demand forecasting', 'forecasts demand', 'sales forecasting', 'forecasts sales', 'revenue forecasting', 'forecasts revenue', 'trend forecasting', 'forecasts trends', 'seasonal forecasting', 'load forecasting', 'forecasts load', 'energy forecasting', 'forecasts energy demand', 'stock forecasting', 'forecasts stock prices', 'price forecasting', 'forecasts prices')
EXPECTED_NOFORECAST = ('no forecasting', 'does not forecast', 'without forecasting', 'no time series forecasting', 'does not forecast time series', 'forecasting not supported', 'does not support forecasting', 'no future predictions', 'does not predict the future', 'no demand forecasting', 'does not forecast demand', 'cannot forecast', 'never forecasts', 'no sales forecasting', 'does not forecast sales', 'no trend forecasting', 'does not predict future values', 'no predictions', 'prediction not supported', 'does not support prediction', 'historical data only', 'descriptive statistics only', 'no forward looking predictions', 'past data only')


class AuditTests(unittest.TestCase):
    def fixture(self, source=None, text='time series forecasting', filename='app.py'):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / 'README.md').write_text(text)
        if source is not None:
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        return root

    def cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(list(map(str, args)))
        return code, out.getvalue()

    def test_phrase_oracle(self):
        tree = ast.parse(Path(phrases.__file__).read_text())
        actual = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)}
        self.assertEqual(actual['FORECAST'], EXPECTED_FORECAST)
        self.assertEqual(actual['NOFORECAST'], EXPECTED_NOFORECAST)
        self.assertEqual(len(EXPECTED_FORECAST), 50)
        self.assertEqual(len(EXPECTED_NOFORECAST), 24)

    def test_all_phrases(self):
        for phrase in EXPECTED_FORECAST:
            with self.subTest(phrase=phrase):
                self.assertEqual(claims(phrase), (True, False))
        for phrase in EXPECTED_NOFORECAST:
            with self.subTest(phrase=phrase):
                self.assertEqual(claims(phrase), (False, True))

    def test_normalization_and_boundaries(self):
        self.assertEqual(claims("Doesn't forecast time-series"), (False, True))
        self.assertEqual(claims('TIME-SERIES   FORECASTING'), (True, False))
        self.assertEqual(claims('xtime series forecastingx'), (False, False))
        self.assertEqual(claims('time series\n\nforecasting'), (False, False))
        self.assertEqual(claims('no forecasting\n\ntime series forecasting'), (True, True))

    def test_fences_inline(self):
        for fence in ['```', '~~~', '   ````']:
            self.assertEqual(claims(fence + '\n time series forecasting\n' + fence), (False, False))
        self.assertEqual(claims('`time series forecasting`'), (True, False))

    def test_routes(self):
        sources = [
            'from statsmodels.tsa.arima.model import ARIMA\nm=ARIMA(y)\nr=m.fit()\nr.forecast(5)',
            'from statsmodels.tsa.statespace.sarimax import SARIMAX\nm=SARIMAX(y)\nm.predict()',
            'from statsmodels.tsa.holtwinters import ExponentialSmoothing\nExponentialSmoothing(y).fit().forecast(5)',
            'from prophet import Prophet\nm=Prophet()\nm.predict(future)',
            'import pmdarima as pm\nm=pm.auto_arima(y)\nm.predict(5)',
            'from neuralforecast import NeuralForecast\nm=NeuralForecast(models=[])\nm.predict()',
            'from darts.models import NBEATSModel\nm=NBEATSModel()\nm.predict(5)',
            'from sktime.forecasting.theta import ThetaForecaster\nm=ThetaForecaster()\nm.predict(fh)',
            'x.forecast(5)',
            'from custom import forecast as fc\nfc(y)',
            'from statsmodels.tsa.arima.model import ARIMA as A\nm=A(y)\nm.get_prediction()',
            'import statsmodels as sm\nm=sm.tsa.arima.model.ARIMA(y)\nm.predict()',
            'import prophet as pp\npp.Prophet().fit(df).predict(future)',
        ]
        for source in sources:
            with self.subTest(source=source):
                root = self.fixture(source)
                self.assertTrue(evidence(source))
                self.assertEqual(self.cli('audit', root)[0], 0)
                self.assertEqual(audit(root)['findings'], [])

    def test_all_route_models_and_fit_chains(self):
        for module, names, methods in ROUTES:
            for name in names:
                for method in methods:
                    for receiver in ['m', 'm.fit()', name + '().fit()']:
                        source = 'from ' + module + ' import ' + name + '\nm=' + name + '()\n' + receiver + '.' + method + '()'
                        with self.subTest(source=source):
                            self.assertTrue(evidence(source))

    def test_all_generic(self):
        for name in GENERIC:
            self.assertTrue(evidence('x.' + name + '()'))
            self.assertTrue(evidence('from lib import ' + name + ' as f\nf()'))
            self.assertFalse(evidence(name + '()'))
            self.assertFalse(evidence('def ' + name + '(): pass\nx.' + name + '()'))

    def test_not_evidence(self):
        sources = [
            'import statsmodels.tsa; import prophet; import pmdarima; import neuralforecast; import darts; import sktime',
            'from statsmodels.tsa.arima.model import ARIMA\nm=ARIMA(y)',
            'from statsmodels.tsa.arima.model import ARIMA\nm=ARIMA(y)\nm.fit()',
            'from prophet import Prophet\nm=Prophet()\nm.make_future_dataframe(periods=5)',
            'model.backtest(); cross_validate(model); sliding_window_split(y); evaluate(model)',
            'mae(y,p); mape(y,p); rmse(y,p); smape(y,p)',
            'pandas.read_csv(path); numpy.array(y); open(path).read()',
            'plt.plot(y); sns.lineplot(y); plotly.line(y)',
            'model.predict(); model.get_prediction(); model.train()',
            'from sklearn.linear_model import LinearRegression\nm=LinearRegression()\nm.fit(x,y)\nm.predict(x)',
            'subprocess.run(["Rscript", "forecast.R"]); subprocess.run(["prophet"])',
            '"x.forecast()" # x.forecast()',
            'x.forecast',
            'x.forecast(',
            'from prophet import Prophet\ndef Prophet(): pass\nm=Prophet()\nm.predict()',
            'from custom import forecast as f\ndef f(): pass\nf()',
            'class Thing:\n def forecast(self): pass\nx.forecast()',
            'from prophet import Prophet as P\ndef Prophet(): pass\nm=P()\nm.predict()',
            'ARIMA(y).predict()',
            'from prophet import Prophet; Prophet().predict(future)',
            'from prophet import Prophet; m=Prophet().fit(df); m.predict(future)',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertFalse(evidence(source))

    def test_bad_and_json(self):
        root = self.fixture('pass')
        code, output = self.cli('audit', root, '--json')
        self.assertEqual(code, 2)
        result = json.loads(output)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['findings'][0]['rule'], 'documented-time-series-forecasting-claim-false')
        self.assertEqual(result['findings'][0]['severity'], 'HIGH')
        self.assertTrue(result['findings'][0]['message'])

    def test_test_only(self):
        for filename in ['tests/evidence.py', 'test_evidence.py', 'evidence_test.py']:
            root = self.fixture('x.forecast()', filename=filename)
            self.assertEqual(audit(root)['findings'][0]['severity'], 'MEDIUM')
            self.assertEqual(audit(root)['findings'][0]['rule'], 'documented-time-series-forecasting-claim-test-only')
            self.assertEqual(self.cli('audit', root)[0], 0)
            self.assertEqual(self.cli('audit', root, '--fail-on', 'medium')[0], 2)
            self.assertEqual(self.cli('audit', root, '--fail-on', 'MEDIUM')[0], 2)

    def test_denial_and_no_claim(self):
        root = self.fixture('x.forecast()', 'no forecasting')
        self.assertEqual(self.cli('audit', root)[0], 2)
        self.assertEqual(audit(root)['findings'][0]['rule'], 'documented-no-time-series-forecasting-claim-false')
        self.assertEqual(audit(root)['findings'][0]['severity'], 'HIGH')
        self.assertEqual(audit(self.fixture('x.forecast()', 'A utility.'))['verdict'], 'PASS')

    def test_no_source(self):
        self.assertEqual(self.cli('audit', self.fixture())[0], 2)
        self.assertEqual(self.cli('audit', self.fixture(text='no forecasting'))[0], 0)

    def test_split_modules(self):
        root = self.fixture('from prophet import Prophet\nm=Prophet()')
        (root / 'other.py').write_text('m.predict()')
        self.assertEqual(self.cli('audit', root)[0], 2)

    def test_excluded_paths(self):
        for folder in ['fixtures', 'docs', 'examples']:
            root = self.fixture('x.forecast()', filename=folder + '/app.py')
            self.assertEqual(self.cli('audit', root)[0], 2)

    def test_symlinks(self):
        root = self.fixture()
        outside = self.fixture('x.forecast()')
        (root / 'linked.py').symlink_to(outside / 'app.py')
        (root / 'linked').symlink_to(outside, target_is_directory=True)
        self.assertEqual(self.cli('audit', root)[0], 2)

    def test_docs_selection_and_ignore(self):
        root = self.fixture('x.forecast()', 'A utility.')
        (root / 'other.md').write_text('no forecasting')
        self.assertEqual(audit(root)['verdict'], 'PASS')
        (root / 'nested/docs').mkdir(parents=True)
        (root / 'nested/docs/guide.md').write_text('no forecasting')
        self.assertEqual(self.cli('audit', root)[0], 2)
        self.assertEqual(self.cli('audit', root, '--ignore', 'nested', '--ignore', 'app.py')[0], 0)
        self.assertIn('Production evidence:', self.cli('audit', root, '--verbose')[1])

    def test_errors(self):
        root = self.fixture()
        self.assertEqual(self.cli('audit', root / 'missing')[0], 1)
        self.assertEqual(self.cli('audit', root / 'README.md')[0], 1)
        for args in [[], ['audit', '--unknown'], ['audit', '--fail-on', 'low'], ['audit', '--ignore']]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    main(args)
                self.assertEqual(exc.exception.code, 1)
        (root / 'bad.py').write_bytes(b'\xff')
        self.assertEqual(self.cli('audit', root)[0], 1)

    def test_entry_points(self):
        root = self.fixture('x.forecast()')
        result = subprocess.run([sys.executable, '-m', 'time_series_forecasting_claim_audit', '--version'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), '0.1.0')
        result = subprocess.run([sys.executable, '-m', 'time_series_forecasting_claim_audit', 'audit', str(root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), 'PASS')
        project = Path(__file__).resolve().parents[1]
        if (project / 'pyproject.toml').exists():
            self.assertIn('time-series-forecasting-claim-audit = "time_series_forecasting_claim_audit.audit:main"', (project / 'pyproject.toml').read_text())


if __name__ == '__main__':
    unittest.main()
