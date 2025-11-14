import pandas as pd
import os
import zipfile
import shutil
import json

class CsvManager():
    def __init__(self, csvFilePath, columns):
        self.csvFilePath = csvFilePath
        self.columns = columns
        if not os.path.exists(self.csvFilePath):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.csvFilePath, index=False)
            print(f"Created the following csv file: {self.csvFilePath}")
    
    def loadDataset(self):
        df = pd.read_csv(self.csvFilePath)
        return df

    def saveDataset(self, df):
        df.to_csv(self.csvFilePath, index=False)
    
    def getColumn(self, columnName):
        df = self.loadDataset()
        return df[columnName].values.tolist()

    def getRow(self, key, df=None):
        df = self.loadDataset()
        return df.loc[df[self.columns[0]] == key]
    
    def getValue(self, valueColumn, key):
        df = self.loadDataset()
        rowIndex = df.loc[df[self.columns[0]] == key].index
        return df.loc[rowIndex, valueColumn].values[0]

    def modifyValue(self, key, newValueColumn, newValue):
        df = self.loadDataset()
        rowIndex = df.loc[df[self.columns[0]] == key].index
        df.loc[rowIndex, newValueColumn] = newValue
        self.saveDataset(df=df)

    def removeRow(self, key):
        df = self.loadDataset()
        df.drop(df[df[self.columns[0]] == key].index, inplace=True)
        self.saveDataset(df=df)
    
    def appendRow(self, values):
        if len(values) != len(self.columns):
            raise ValueError("list of values when appending a new row should be the same length of the columns in the dataframe")
        
        df = self.loadDataset()
        df.loc[len(df)] = values
        self.saveDataset(df=df)

class ZipManager():
    def __init__(self):
        pass

    def zipDir(self, newZipFilePath, dirPath, removePath=False):
        shutil.make_archive(newZipFilePath, 'zip', dirPath)
        if removePath:
            shutil.rmtree(dirPath)

    def unzipFile(self, zipFilePath, destinationDir, removeArchive=False):
        with zipfile.ZipFile(zipFilePath, "r") as zip:
            zip.extractall(destinationDir)
        if removeArchive:
            os.remove(zipFilePath)
    
    def removeFileFromZip(self, zipFilePath, tempDirPath, fileName):
        if not os.path.exists(tempDirPath):
            os.mkdir(tempDirPath)
        self.unzipFile(zipFilePath=zipFilePath, destinationDir=tempDirPath, removeArchive=True)
        for file in os.listdir(tempDirPath):
            if file == fileName:
                os.remove(os.path.join(tempDirPath, file))
        self.zipDir(newZipFilePath=zipFilePath, dirPath=tempDirPath, removePath=True)
    
    def addFileToZip(self, zipFilePath, addFilePath):
        with zipfile.ZipFile(zipFilePath, "a") as zip:
            zip.write(addFilePath)

    def getZipNameList(self, zipFilePath):
        with zipfile.ZipFile(zipFilePath, 'r') as zip:
            return zip.namelist()

class JsonManager():
    def __init__(self):
        pass

    def loadJson(self, jsonPath):
        with open(jsonPath, "r", encoding="utf8") as jsonFile: 
            jsonDict = json.load(jsonFile)
            return jsonDict
    
    def saveJson(self, jsonPath, jsonDict={}, indent=2):
        with open(jsonPath, "w", encoding="utf8") as jsonFile:
            json.dump(jsonDict, jsonFile, indent=indent)
    
    def getKeys(self, jsonPath):
        jsonDict = self.loadJson(jsonPath)
        return list(jsonDict.keys())
    
    def getValue(self, jsonPath, key):
        jsonDict = self.loadJson(jsonPath)
        return jsonDict[key]
    
    def getValues(self, jsonPath):
        jsonDict = self.loadJson(jsonPath)
        values = []
        for key in self.getKeys(jsonPath):
            values.append(jsonDict[key])
        return values

    def modifyValue(self, jsonPath, key, newValue):
        jsonDict = self.loadJson(jsonPath)
        jsonDict[key] = newValue
        self.saveJson(jsonPath, jsonDict)
    
    def modifyValues(self, jsonPath, keyList, valueList):
        jsonDict = self.loadJson(jsonPath)
        for key, value in zip(keyList, valueList):
            jsonDict[key] = value
        self.saveJson(jsonPath, jsonDict)
    
    def removeValue(self, jsonPath, key):
        jsonDict = self.loadJson(jsonPath)
        jsonDict.pop(key)
        self.saveJson(jsonPath, jsonDict)

    def removeValues(self, jsonPath, keyList):
        for key in keyList:
            self.removeValue(jsonPath, key)
            
class txtManager():
    def __init__(self, debugMessage=False, encoding="utf8"):
        self.debugMode = debugMessage
        self.encoding = encoding
    
    def createEmpty(self, txtFilePath, overWrite=True):
        if not overWrite:
            if os.path.exists(txtFilePath):
                return
        with open(txtFilePath, "w", encoding=self.encoding):
            pass

    def writeText(self, txtFilePath, text, append=False):
        if append:
            mode="a"
        else:
            mode="w"
        with open(txtFilePath, mode, encoding=self.encoding) as f:
            f.write(text)  
    
    def getText(self, txtFilePath):
        with open(txtFilePath, "r", encoding=self.encoding) as f:
            return f.read()
    
    def getLines(self, txtFilePath):
        with open(txtFilePath, "r", encoding=self.encoding) as f:
            return [line.strip() for line in f.readlines()]

    def removeChar(self, txtFilePath, charToReplace, replaceChar="", saveFile=True):
        fileText = self.getText(txtFilePath=txtFilePath)
        fileText = fileText.replace(charToReplace, replaceChar)
        if saveFile:
            self.writeText(txtFilePath=txtFilePath, text=fileText)
        return fileText  