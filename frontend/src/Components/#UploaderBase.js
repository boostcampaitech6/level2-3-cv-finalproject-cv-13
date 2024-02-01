import {useEffect, useState} from 'react';
import "./uploader.scss";
import {Button} from "@mui/material";
import HighlightOffIcon from '@mui/icons-material/HighlightOff';

const Uploader = () => {
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
    if (files) {
      for (let i = 0; i < files.length; i++) {
        const preview_URL = URL.createObjectURL(files[i]);
        const fileType = files[i].type.split("/")[0];
        fileList.push({
        fileObject: files[i],
        preview_URL: preview_URL,
        type: fileType,
          });
      }
    }
    // 마지막에 state update
    setFileList([...tmpFileList, ...fileList]);
  };
  // index에 해당하는 state 삭제
  const deleteImage = (index) => {
    const tmpFileList = [...fileList];
    tmpFileList.splice(index, 1);
    setFileList(tmpFileList);
  }

  useEffect(()=>{
    // 컴포넌트가 언마운트되면 revokeObjectURL로 미리보기 URL 모두 삭제
    return () => {
      fileList?.forEach((item)=>{
        URL.revokeObjectURL(item.preview_URL);
      })
    }
  }, [])

  console.log(fileList);
  return (
    <div className="uploader-wrapper">
      <input
        type="file" multiple={true} accept="video/*, image/*"
        onChange={saveImage}
        // 클릭할 때 마다 file input의 value를 초기화 하지 않으면 버그가 발생할 수 있다
        // 사진 등록을 두개 띄우고 첫번째에 사진을 올리고 지우고 두번째에 같은 사진을 올리면 그 값이 남아있음!
        onClick={(e) => e.target.value = null}
        ref={refParam => inputRef = refParam}
        style={{display: "none"}}
      />
      <div className="file-container">
          {fileList?.map((item, index) => (
            <div className="file-wrapper">
              {item.type === "image" ? (
                <img src={item.preview_URL}/>

              ) : (
                <video src={item.preview_URL} autoPlay={false} controls={true}/>
              )}
              <div className="delete-button" onClick={()=>{deleteImage(index)}}>
                <HighlightOffIcon fontSize="large" color="error"/>
              </div>
            </div>
          ))}
      </div>
      <div className="upload-button">
        <Button variant="contained" onClick={() => inputRef.click()}>
          Upload
        </Button>
      </div>
    </div>
  );
}

export default Uploader;