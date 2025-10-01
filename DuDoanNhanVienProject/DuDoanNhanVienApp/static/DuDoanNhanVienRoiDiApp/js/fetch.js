const api = "http://127.0.0.1:8000/";
export async function fetchAPI(action = "", method = "GET", data = null, parram) {
    try {
        const url = `${api} + ${action}`;
        const option = {
            Headers: {
                "Content-Type": "application/json",
            },
            method,
        }
        if(data !== null && method !== 'GET') {
            option.body = JSON.stringify(data);
        }
        const response = await fetch(url, option);
        if(!response.ok) {
            const resultText = await response.text();
            console.log("loi: ", response.status, ": ", resultText);
            return null;
        }
        const data = await response.json();
        if(!data.error){
            console.log("Thanh Cong: ", data);
            return data;
        }
            console.log("Co loi Xay Ra: ", data.message);
            return null;
    } catch (error) {
        console.log("Cos Loi Xay Ra: ", error.message);
        return;
    }
}