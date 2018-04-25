@echo off

cd C:\Windows\System32\
FOR /f "tokens=*" %%G IN ('dir /b') DO (
	echo %%G
	"C:\Users\chanw\Desktop\NtTrace\NtTrace" %%G > "C:\Users\chanw\Desktop\beLog\LOG_%%G.txt"	
)
