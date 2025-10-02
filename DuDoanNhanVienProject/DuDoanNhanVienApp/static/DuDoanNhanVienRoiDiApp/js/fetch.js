const api = "http://127.0.0.1:8000/";
export async function fetchAPI(action = "", method = "GET", data = null, isFile = false) {
    try {
        const url = `${api}${action}`;
        const option = {
            method,
        }
        if(data !== null && method !== 'GET') {
            option.body = data;
        } 
        console.log(option);
        const response = await fetch(url, option);
        if(!response.ok) {
            const resultText = await response.text();
            console.log("loi: ", response.status, ": ", resultText);
            return null;
        }
        const resultData = await response.json();
        if(!resultData.error){
            console.log("Thanh Cong: ", resultData);
            return resultData.data;
        }   
        console.log("Loi: ", resultData);
        console.log("Co loi Xay Ra: ", resultData.error);
        return null;
    } catch (error) {
        console.log("Co Loi Xay Ra O Fetch: ", error.message);
        return;
    }
}
