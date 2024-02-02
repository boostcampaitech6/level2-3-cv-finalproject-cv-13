import React from 'react'
import axios from 'axios'
import './InputScreen.css'
import ImgAsset from '../public'
// import {Link} from 'react-router-dom'
import {useEffect, useState} from 'react';
import {Button} from "@mui/material";

export default function InputScreen () {
	/*
    fileList는 아래와 같은 object의 array로 구성
    {
      fileObject: files[i],
      preview_URL: preview_URL,
      type: fileType,
    }
  	*/

	const [fileList, setFileList] = useState([]);
  	let inputRef;

	const saveImage = async (e) => {
	e.preventDefault();
	// state update전 임시로 사용할 array
	const tmpFileList = [];
	const files = e.target.files;
	const formData = new FormData();
	if (files) {
		for (let i = 0; i < files.length; i++) {
		const preview_URL = URL.createObjectURL(files[i]);
		const fileType = files[i].type.split("/")[0];
		fileList.push({
		fileObject: files[i],
		preview_URL: preview_URL,
		type: fileType,
			});
		formData.append('file', files[i]);
		formData.append('id', i);
		}

		// fileList.forEach(file=>{
		// 	formData.append("arrayOfFilesName", file);
		//   });
		const postOptions = {
			method: "POST",
			url: "http://127.0.0.1:8000/input",
			body: formData,
		}

		try {
            // fetch를 이용한 post 요청.
            const response = await fetch("http://127.0.0.1:8000/input", postOptions)
            alert('POST 완료');
			console.log(response);
        } catch (err) {
            alert(err || 'POST 실패');
        }
	}
	// 마지막에 state update
	setFileList([...tmpFileList, ...fileList]);
	};

	console.log(fileList);
	
	return (
		<div className='InputScreen_InputScreen'>
			<img className='background' src = {ImgAsset.InputScreen_background} alt="background"/>
			<div className='ImageHolderGroup'>
					<input
					type="file" multiple={true} accept="image/*"
					onChange={saveImage}
					ref={refParam => inputRef = refParam}
					style={{display: "none"}}
					/>
				<div className='imageholder'/>
				<img className='Imagelogo' src = {ImgAsset.InputScreen_Imagelogo} alt="imagelogo"/>
				<span className='Text'>Place images here<br/>or<br/>Upload from your computer</span>
				<div className='UploadButton'>
					<Button variant="contained" onClick={() => inputRef.click()}>
						Upload
					</Button>
				</div>
			</div>
			<img className='logo' src = {ImgAsset.LoadingScreen_logo} alt="logo"/>
		</div>
	)
}