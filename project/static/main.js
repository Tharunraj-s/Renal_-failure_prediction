        let currentPage = 1;
        const form = document.getElementById("multi-page-form");

        function nextPage() {
            if (currentPage < 3) {
                currentPage++;
                updatePage();
            }
        }

        function previousPage() {
            if (currentPage > 1) {
                currentPage--;
                updatePage();
            }
        }

        function updatePage() {
            const pages = document.getElementsByClassName("form-column");
            for (let i = 0; i < pages.length; i++) {
                pages[i].style.display = "none";
            }
            pages[currentPage - 1].style.display = "block";

            if (currentPage === 1) {
                document.getElementById("back-button").style.display = "none";
                document.getElementById("next-button").style.display = "inline";
                document.getElementById("predict-button").style.display = "none";
            } else if (currentPage === 3) {
                document.getElementById("back-button").style.display = "inline";
                document.getElementById("next-button").style.display = "none";
                document.getElementById("predict-button").style.display = "inline";
            } else {
                document.getElementById("back-button").style.display = "inline";
                document.getElementById("next-button").style.display = "inline";
                document.getElementById("predict-button").style.display = "none";
            }
        }

        updatePage(); // Initialize to the first page