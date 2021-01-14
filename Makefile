# Makefile, converts qt5 ui files in base classes for python

all : tres_guider_control_ui.py tres_guider_config_ui.py tres_guider_display_ui.py

tres_guider_control_ui.py : tres_guider_control.ui
	pyuic5 -x tres_guider_control.ui -o tres_guider_control_ui.py

tres_guider_config_ui.py : tres_guider_config.ui
	pyuic5 -x tres_guider_config.ui -o tres_guider_config_ui.py

tres_guider_display_ui.py : tres_guider_display.ui
	pyuic5 -x tres_guider_display.ui -o tres_guider_display_ui.py


clean:
	rm tres_guider_config_ui.py tres_guider_control_ui.py tres_guider_display_ui.py

