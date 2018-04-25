@echo off

cd ./MalFolder
FOR /f "tokens=*" %%G IN ('dir /b') DO (
	"./../NtTrace/NtTrace" %%G > ./../MalLog/LOG_%%G.txt	
)
