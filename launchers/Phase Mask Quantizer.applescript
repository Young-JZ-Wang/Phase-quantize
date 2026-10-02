on run
	set appDir to POSIX path of (do shell script "dirname " & quoted form of POSIX path of (path to me))
	set cmd to "cd " & quoted form of appDir & " && for py in python3 /opt/anaconda3/bin/python3 /usr/local/bin/python3 /opt/homebrew/bin/python3 /usr/bin/python3; do if command -v $py >/dev/null 2>&1 && $py -c 'import numpy, PIL, scipy, tkinter' >/dev/null 2>&1; then nohup $py phase_mask_quantizer.py >/dev/null 2>&1 & exit 0; fi; done; exit 1"
	try
		do shell script "export PATH=/opt/anaconda3/bin:/usr/local/bin:/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin; " & cmd
	on error
		display alert "Phase Mask Quantizer" message "No Python with numpy, pillow, scipy and tkinter was found." & return & "Install with: python3 -m pip install numpy pillow scipy h5py" as critical
	end try
end run
