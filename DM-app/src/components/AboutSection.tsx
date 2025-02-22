import { Container, Header, TextContent } from "@cloudscape-design/components";

export const AboutSection: React.FC = () => {
    return (
        <Container
            header={<Header variant="h2">About</Header>}
        >
            <TextContent>
                <p>Welcome to the <strong>Stable Chess GM Rankings</strong> application! This interactive tool is designed to compute and analyze stable rankings of Grandmaster chess players based on the framework outlined in <i>On Obtaining Stable Rankings</i>  (Asudeh etal., PVLDB 2018).</p>

                <br/>
                <h4>How to Use</h4>
                <ol>
                    <li><strong>Select Attributes</strong>: Choose two scoring attributes from the provided list.</li>
                    <li><strong>Pick an Algorithm</strong>: Select a ranking algorithm that best suits your analysis needs.</li>
                    <ol type="i">
                        <li><strong>Raysweeping</strong>: A deterministic method that utilizes a geometric approach to analyze the dataset in dual space. It works by representing rankings as regions within a weight space, allowing for a visual interpretation of ranking stability.</li>
                        <li><strong>Randomized rounding</strong>: Randomly samples from the weight space to generate rankings, using Bernoulli random variables to influence the selection process. By iteratively refining the selection based on stability measurements, it can efficiently identify stable rankings even in larger, more complex datasets.</li>
                    </ol>
                    <li><strong>Enter Constraints</strong>: Define any constraints for the weight function. Ensure constraints follow this structure: <code>number*w1 operator number*w2</code> (e.g., <code>1*w1{">"}=2*w2</code>), with each constraint on a new line.</li>
                    <li><strong>Submit</strong>: Click the submit button to generate the rankings based on your selections.</li>
                </ol>
            </TextContent>
        </Container>
    );
}