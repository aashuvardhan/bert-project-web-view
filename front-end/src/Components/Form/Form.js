import React, { useState } from 'react'

export const Form=(props)=> {

    const [content,setAddContent]=useState('')
    
    const contentChangeHandler=(event)=>{
        setAddContent(event.target.value)
        
    }

    const submitHandler=(event)=>{
        event.preventDefault()
        
        props.formHandler(content)
        setAddContent('')
    }

    

  return (
    <div>
        Form:-
        <form onSubmit={submitHandler}>
            <input type="text" name="content" value={content} onChange={contentChangeHandler} required/>
            <input type="submit"/>
        </form>
    
    </div>
  )
}
