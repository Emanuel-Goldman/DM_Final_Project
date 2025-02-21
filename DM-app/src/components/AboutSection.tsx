import { Container, Header, TextContent } from "@cloudscape-design/components";

export const AboutSection: React.FC = () => {
    return (
        <Container
            header={<Header variant="h2">About</Header>}
        >
            <TextContent>
                <p>
                    Users can select attributes from the Chess All GM Players Statistics 2023 dataset, apply stability-based ranking algorithms, and explore the most robust rankings. 
                    The app supports ranking stability verification, helping to understand how small changes in weighting affect player rankings.
                </p>
            </TextContent>
        </Container>
    );
}