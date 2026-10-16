# Vendored from chenchen0103/ACEBench@56dd66cf6439 (MIT, see __init__.py):
# model_inference/multi_turn/scenarioszh/phone_platform/base_api.py
# Verbatim. Do not edit.

class BaseApi:
    
    def __init__(self):
        self.wifi = False
        self.logged_in = True
    
    def _load_scenario(self, scenario: dict, long_context=False) -> None:

        self.wifi = scenario.get("wifi", False)
        self.logged_in = scenario.get("logged_in",True)

    
    def turn_on_wifi(self):
        self.wifi = True
        return {'status': True, 'message': 'wifi已经打开'} 
    
    def login_device(self):
        self.logged_in = True
        return {'status': True, 'message': '设备已经登录'} 
        


