
console.log("ABC 12345");


addEventListener("submit", (event) => {
    event.preventDefault();
    console.log(event.target);
    formSubmit = document.getElementById(event.target.id);
    const button = formSubmit.querySelector("button");
    button.hidden = true;
});